/**
 * @license
 * Copyright 2019 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {User} from '../auth/user';
import {
  Bound,
  canonifyTarget,
  FieldFilter,
  Operator,
  Target,
  targetEquals,
  targetGetArrayValues,
  targetGetLowerBound,
  targetGetNotInValues,
  targetGetUpperBound
} from '../core/target';
import {Value as ProtoValue} from '../protos/firestore_proto_api';
import {DocumentKeySet, DocumentMap} from '../model/collections';
import {
  FieldIndex,
  fieldIndexGetDirectionalSegments,
  IndexKind,
  IndexOffset,
  IndexSegment
} from '../model/field_index';
import {FieldPath, ResourcePath} from '../model/path';
import {debugAssert, fail} from '../util/assert';
import {immediateSuccessor} from '../util/misc';

import {decodeResourcePath, encodeResourcePath} from './encoded_resource_path';
import {IndexManager} from './index_manager';
import {
  DbCollectionParent,
  DbCollectionParentKey,
  DbIndexConfiguration,
  DbIndexConfigurationKey,
  DbIndexEntry,
  DbIndexEntryKey,
  DbIndexState,
  DbIndexStateKey
} from './indexeddb_schema';
import {getStore} from './indexeddb_transaction';
import {
  fromDbIndexConfiguration,
  toDbIndexConfiguration,
  toDbIndexState
} from './local_serializer';
import {MemoryCollectionParentIndex} from './memory_index_manager';
import {PersistencePromise} from './persistence_promise';
import {PersistenceTransaction} from './persistence_transaction';
import {SimpleDbStore} from './simple_db';
import {logDebug} from '../util/log';
import {ObjectMap} from '../util/obj_map';
import {Document} from '../model/document';
import {IndexByteEncoder} from '../index/index_byte_encoder';
import {FirestoreIndexValueWriter} from '../index/firestore_index_value_writer';
import {isArray} from '../model/values';

const LOG_TAG = 'IndexedDbIndexManager';

/**
 * A persisted implementation of IndexManager.
 *
 * PORTING NOTE: Unlike iOS and Android, the Web SDK does not memoize index
 * data as it supports multi-tab access.
 */
export class IndexedDbIndexManager implements IndexManager {
  /**
   * An in-memory copy of the index entries we've already written since the SDK
   * launched. Used to avoid re-writing the same entry repeatedly.
   *
   * This is *NOT* a complete cache of what's in persistence and so can never be
   * used to satisfy reads.
   */
  private collectionParentsCache = new MemoryCollectionParentIndex();

  private uid: string;

  /**
   * Maps from a target to its equivalent list of sub-targets. Each sub-target
   * contains only one term from the target's disjunctive normal form (DNF).
   */
  private targetToDnfSubTargets = new ObjectMap<Target, Target[]>(
    t => canonifyTarget(t),
    (l, r) => targetEquals(l, r)
  );

  constructor(private user: User) {
    this.uid = user.uid || '';
  }

  /**
   * Adds a new entry to the collection parent index.
   *
   * Repeated calls for the same collectionPath should be avoided within a
   * transaction as IndexedDbIndexManager only caches writes once a transaction
   * has been committed.
   */
  addToCollectionParentIndex(
    transaction: PersistenceTransaction,
    collectionPath: ResourcePath
  ): PersistencePromise<void> {
    debugAssert(collectionPath.length % 2 === 1, 'Expected a collection path.');
    if (!this.collectionParentsCache.has(collectionPath)) {
      const collectionId = collectionPath.lastSegment();
      const parentPath = collectionPath.popLast();

      transaction.addOnCommittedListener(() => {
        // Add the collection to the in memory cache only if the transaction was
        // successfully committed.
        this.collectionParentsCache.add(collectionPath);
      });

      const collectionParent: DbCollectionParent = {
        collectionId,
        parent: encodeResourcePath(parentPath)
      };
      return collectionParentsStore(transaction).put(collectionParent);
    }
    return PersistencePromise.resolve();
  }

  getCollectionParents(
    transaction: PersistenceTransaction,
    collectionId: string
  ): PersistencePromise<ResourcePath[]> {
    const parentPaths = [] as ResourcePath[];
    const range = IDBKeyRange.bound(
      [collectionId, ''],
      [immediateSuccessor(collectionId), ''],
      /*lowerOpen=*/ false,
      /*upperOpen=*/ true
    );
    return collectionParentsStore(transaction)
      .loadAll(range)
      .next(entries => {
        for (const entry of entries) {
          // This collectionId guard shouldn't be necessary (and isn't as long
          // as we're running in a real browser), but there's a bug in
          // indexeddbshim that breaks our range in our tests running in node:
          // https://github.com/axemclion/IndexedDBShim/issues/334
          if (entry.collectionId !== collectionId) {
            break;
          }
          parentPaths.push(decodeResourcePath(entry.parent));
        }
        return parentPaths;
      });
  }

  addFieldIndex(
    transaction: PersistenceTransaction,
    index: FieldIndex
  ): PersistencePromise<void> {
    // TODO(indexing): Verify that the auto-incrementing index ID works in
    // Safari & Firefox.
    const indexes = indexConfigurationStore(transaction);
    const dbIndex = toDbIndexConfiguration(index);
    delete dbIndex.indexId; // `indexId` is auto-populated by IndexedDb
    return indexes.add(dbIndex).next();
  }

  deleteFieldIndex(
    transaction: PersistenceTransaction,
    index: FieldIndex
  ): PersistencePromise<void> {
    const indexes = indexConfigurationStore(transaction);
    const states = indexStateStore(transaction);
    const entries = indexEntriesStore(transaction);
    return indexes
      .delete(index.indexId)
      .next(() =>
        states.delete(
          IDBKeyRange.bound(
            [index.indexId],
            [index.indexId + 1],
            /*lowerOpen=*/ false,
            /*upperOpen=*/ true
          )
        )
      )
      .next(() =>
        entries.delete(
          IDBKeyRange.bound(
            [index.indexId],
            [index.indexId + 1],
            /*lowerOpen=*/ false,
            /*upperOpen=*/ true
          )
        )
      );
  }

  getDocumentsMatchingTarget(
    transaction: PersistenceTransaction,
    target: Target
  ): PersistencePromise<DocumentKeySet> {
    const subQueries: string[] = [];
    const bindings: unknown[] = [];

    for (const subTarget of this.getSubTargets(target)) {
      const fieldIndex = this.getFieldIndex(subTarget);
      const arrayValues = targetGetArrayValues(subTarget, fieldIndex);
      const notInValues = targetGetNotInValues(subTarget, fieldIndex);
      const lowerBound = targetGetLowerBound(subTarget, fieldIndex);
      const upperBound = targetGetUpperBound(subTarget, fieldIndex);

      logDebug(
        LOG_TAG,
        "Using index '%s' to execute '%s' (Arrays: %s, Lower bound: %s, Upper bound: %s)",
        fieldIndex,
        subTarget,
        arrayValues,
        lowerBound,
        upperBound
      );

      const lowerBoundEncoded = this.encodeBound(
        fieldIndex,
        subTarget,
        lowerBound
      );
      const lowerBoundOp =
        lowerBound != null && lowerBound.isInclusive() ? '>=' : '>';
      const upperBoundEncoded = this.encodeBound(
        fieldIndex,
        subTarget,
        upperBound
      );
      const upperBoundOp =
        upperBound != null && upperBound.isInclusive() ? '<=' : '<';
      const notInEncoded = this.encodeValues(
        fieldIndex,
        subTarget,
        notInValues
      );

      const subQueryAndBindings = this.generateQueryAndBindings(
        subTarget,
        fieldIndex.getIndexId(),
        arrayValues,
        lowerBoundEncoded,
        lowerBoundOp,
        upperBoundEncoded,
        upperBoundOp,
        notInEncoded
      );
      subQueries.push(String.valueOf(subQueryAndBindings[0]));
      bindings.push(
        Arrays.asList(subQueryAndBindings).subList(
          1,
          subQueryAndBindings.length
        )
      );
    }

    let queryString: string;
    if (subQueries.length === 1) {
      // If there's only one subQuery, just execute the one subQuery.
      queryString = subQueries.get(0);
    } else {
      // Construct "SELECT * FROM (subQuery1 UNION subQuery2 UNION ...) LIMIT N"
      queryString =
        'SELECT * FROM (' + TextUtils.join(' UNION ', subQueries) + ')';
      if (target.getLimit() != -1) {
        queryString = queryString + ' LIMIT ' + target.getLimit();
      }
    }

    debugAssert(
      bindings.size() < 1000,
      'Cannot perform query with more than 999 bind elements'
    );

    // SQLitePersistence.Query query = db.query(queryString).binding(bindings.toArray());
    //
    // Set<DocumentKey> result = new HashSet<>();
    // query.forEach(
    //   row -> result.add(DocumentKey.fromPath(ResourcePath.fromString(row.getString(0)))));
    //
    // Logger.debug(TAG, "Index scan returned %s documents", result.size());
    // return result;
  }

  private getSubTargets(target: Target): Target[] {
    let subTargets = this.targetToDnfSubTargets.get(target);
    if (subTargets) {
      return subTargets;
    }
    subTargets = [];
    if (target.filters.length === 0) {
      subTargets.push(target);
    } else {
      // TODO(orquery): Implement DNF transform
      fail('DNF transform not impplemented');
    }
    this.targetToDnfSubTargets.set(target, subTargets);
    return subTargets;
  }

  /** Constructs a key range query on 'index_entries' that unions all bounds.  */
  private generateQueryAndBindings(
    target: Target,
    indexId: number,
    arrayValues: ProtoValue[],
    lowerBounds: unknown[],
    lowerBoundOp: string,
    upperBounds: unknown[] | undefined,
    upperBoundO: string,
    notIn: unknown[] | undefined
  ): IDBKeyRange[] {
    // The number of total statements we union together. This is similar to a
    // distributed normal form, but adapted for array values. We create a single
    // statement per value in an ARRAY_CONTAINS or ARRAY_CONTAINS_ANY filter
    // combined with the values from the query bounds.
    const statementCount =
      (arrayValues != null ? arrayValues.length : 1) *
      max(
        lowerBounds != null ? lowerBounds.length : 1,
        upperBounds != null ? upperBounds.length : 1
      );

      // Build the statement. We always include the lower bound, and optionally
      // include an array value and an upper bound.
      StringBuilder statement = new StringBuilder();
      statement.append("SELECT document_key, directional_value FROM index_entries ");
      statement.append("WHERE index_id = ? AND uid = ? ");
      if (arrayValues != null) {
      statement.append("AND array_value = ? ");
    }
    if (lowerBounds != null) {
      statement.append("AND directional_value ").append(lowerBoundOp).append(" ? ");
    }
    if (upperBounds != null) {
      statement.append("AND directional_value ").append(upperBoundOp).append(" ? ");
    }

    // Create the UNION statement by repeating the above generated statement. We
    // can then add ordering and a limit clause.
    StringBuilder sql = repeatSequence(statement, statementCount, " UNION ");
    sql.append(" ORDER BY directional_value, document_key ");
    if (target.getLimit() != -1) {
      sql.append("LIMIT ").append(target.getLimit()).append(" ");
    }

    if (notIn != null) {
      // Wrap the statement in a NOT-IN call.
      sql = new StringBuilder("SELECT document_key, directional_value FROM (").append(sql);
      sql.append(") WHERE directional_value NOT IN (");
      sql.append(repeatSequence("?", notIn.length, ", "));
      sql.append(")");
    }

    // Fill in the bind ("question marks") variables.
    Object[] bindArgs =
      fillBounds(statementCount, indexId, arrayValues, lowerBounds, upperBounds, notIn);

    List<Object> result = new ArrayList<Object>();
    result.add(sql.toString());
    result.addAll(Arrays.asList(bindArgs));
    return result.toArray();
  }
  //
  // /** Returns the bind arguments for all {@code statementCount} statements. */
  // private  fillBounds(
  //   int statementCount,
  //   int indexId,
  // @Nullable List<Value> arrayValues,
  // @Nullable Object[] lowerBounds,
  // @Nullable Object[] upperBounds,
  // @Nullable Object[] notInValues) : unknown[] {
  //   int bindsPerStatement =
  //     2
  //     + (arrayValues != null ? 1 : 0)
  //     + (lowerBounds != null ? 1 : 0)
  //     + (upperBounds != null ? 1 : 0);
  //   int statementsPerArrayValue = statementCount / (arrayValues != null ? arrayValues.size() : 1);
  //
  //   Object[] bindArgs =
  //     new Object
  //       [statementCount * bindsPerStatement + (notInValues != null ? notInValues.length : 0)];
  //   int offset = 0;
  //   for (int i = 0; i < statementCount; ++i) {
  //     bindArgs[offset++] = indexId;
  //     bindArgs[offset++] = uid;
  //     if (arrayValues != null) {
  //       bindArgs[offset++] = encodeSingleElement(arrayValues.get(i / statementsPerArrayValue));
  //     }
  //     if (lowerBounds != null) {
  //       bindArgs[offset++] = lowerBounds[i % statementsPerArrayValue];
  //     }
  //     if (upperBounds != null) {
  //       bindArgs[offset++] = upperBounds[i % statementsPerArrayValue];
  //     }
  //   }
  //   if (notInValues != null) {
  //     for (Object notInValue : notInValues) {
  //       bindArgs[offset++] = notInValue;
  //     }
  //   }
  //   return bindArgs;
  // }
  //
  //   getFieldIndex(
  //     transaction: PersistenceTransaction,
  //     target: Target
  //   ): PersistencePromise<FieldIndex | null> {
  //
  //   TargetIndexMatcher targetIndexMatcher = new TargetIndexMatcher(target);
  //   String collectionGroup =
  //     target.getCollectionGroup() != null
  //       ? target.getCollectionGroup()
  //       : target.getPath().getLastSegment();
  //
  //   Collection<FieldIndex> collectionIndexes = getFieldIndexes(collectionGroup);
  //   if (collectionIndexes.isEmpty()) {
  //   return null;
  // }
  //
  // List<FieldIndex> matchingIndexes = new ArrayList<>();
  // for (FieldIndex fieldIndex : collectionIndexes) {
  //   boolean matches = targetIndexMatcher.servedByIndex(fieldIndex);
  //   if (matches) {
  //     matchingIndexes.add(fieldIndex);
  //   }
  // }
  //
  // if (matchingIndexes.isEmpty()) {
  //   return null;
  // }
  //
  // // Return the index with the most number of segments
  // return Collections.max(
  //   matchingIndexes, (l, r) -> Integer.compare(l.getSegments().size(), r.getSegments().size()));
  //   }
  //

  /**
   * Returns the byte encoded form of the directional values in the field index.
   * Returns `null` if the document does not have all fields specified in the
   * index.
   */
  private encodeDirectionalElements(
    fieldIndex: FieldIndex,
    document: Document
  ): Uint8Array | null {
    const encoder = new IndexByteEncoder();
    for (const segment of fieldIndexGetDirectionalSegments(fieldIndex)) {
      const field = document.data.field(segment.fieldPath);
      if (field == null) {
        return null;
      }
      const directionalEncoder = encoder.forKind(segment.kind);
      FirestoreIndexValueWriter.INSTANCE.writeIndexValue(
        field,
        directionalEncoder
      );
    }
    return encoder.encodedBytes();
  }

  /** Encodes a single value to the ascending index format. */
  private encodeSingleElement(value: ProtoValue): Uint8Array {
    const encoder = new IndexByteEncoder();
    FirestoreIndexValueWriter.INSTANCE.writeIndexValue(
      value,
      encoder.forKind(IndexKind.ASCENDING)
    );
    return encoder.encodedBytes();
  }

  /**
   * Encodes the given field values according to the specification in `target`.
   * For IN queries, a list of possible values is returned.
   */
  private encodeValues(
    fieldIndex: FieldIndex,
    target: Target,
    bound: Bound | null
  ) {
    if (bound == null) return null;

    let encoders: IndexByteEncoder[] = [];
    encoders.push(new IndexByteEncoder());

    let boundIdx = 0;
    for (const segment of fieldIndexGetDirectionalSegments(fieldIndex)) {
      const value = bound.position[boundIdx++];
      for (const encoder of encoders) {
        if (this.isInFilter(target, segment.fieldPath) && isArray(value)) {
          encoders = this.expandIndexValues(encoders, segment, value);
        } else {
          const directionalEncoder = encoder.forKind(segment.kind);
          FirestoreIndexValueWriter.INSTANCE.writeIndexValue(
            value,
            directionalEncoder
          );
        }
      }
    }
    return this.getEncodedBytes(encoders);
  }

  /**
   * Encodes the given bounds according to the specification in `target`. For IN
   * queries, a list of possible values is returned.
   */
  private encodeBound(
    fieldIndex: FieldIndex,
    target: Target,
    bound: Bound | null
  ) {
    if (bound == null) return null;
    return this.encodeValues(fieldIndex, target, bound);
  }

  /** Returns the byte representation for all encoders. */
  private getEncodedBytes(encoders: IndexByteEncoder[]): Uint8Array[] {
    const result: Uint8Array[] = [];
    for (let i = 0; i < encoders.length; ++i) {
      result[i] = encoders[i].encodedBytes();
    }
    return result;
  }

  /**
   * Creates a separate encoder for each element of an array.
   *
   * The method appends each value to all existing encoders (e.g. filter("a",
   * "==", "a1").filter("b", "in", ["b1", "b2"]) becomes ["a1,b1", "a1,b2"]). A
   * list of new encoders is returned.
   */
  private expandIndexValues(
    encoders: IndexByteEncoder[],
    segment: IndexSegment,
    value: ProtoValue
  ): IndexByteEncoder[] {
    const prefixes = [...encoders];
    const results: IndexByteEncoder[] = [];
    for (const arrayElement of value.arrayValue!.values || []) {
      for (const prefix of prefixes) {
        const clonedEncoder = new IndexByteEncoder();
        clonedEncoder.seed(prefix.encodedBytes());
        FirestoreIndexValueWriter.INSTANCE.writeIndexValue(
          arrayElement,
          clonedEncoder.forKind(segment.kind)
        );
        results.push(clonedEncoder);
      }
    }
    return results;
  }

  private isInFilter(target: Target, fieldPath: FieldPath): boolean {
    for (const filter of target.filters) {
      if (
        filter instanceof FieldFilter &&
        filter.field.isEqual(fieldPath)
      ) {
        return (
          filter.op === Operator.IN ||
          filter.op === Operator.NOT_IN
        );
      }
    }
    return false;
  }

  private encodeSegments(fieldIndex: FieldIndex): Uint8Array {
    return serializer
      .encodeFieldIndexSegments(fieldIndex.getSegments())
      .toByteArray();
  }

  getFieldIndexes(
    transaction: PersistenceTransaction,
    collectionGroup?: string
  ): PersistencePromise<FieldIndex[]> {
    const indexes = indexConfigurationStore(transaction);
    const states = indexStateStore(transaction);

    return (
      collectionGroup
        ? indexes.loadAll(
            DbIndexConfiguration.collectionGroupIndex,
            IDBKeyRange.bound(collectionGroup, collectionGroup)
          )
        : indexes.loadAll()
    ).next(indexConfigs => {
      const result: FieldIndex[] = [];
      return PersistencePromise.forEach(
        indexConfigs,
        (indexConfig: DbIndexConfiguration) => {
          return states
            .get([indexConfig.indexId!, this.uid])
            .next(indexState => {
              result.push(fromDbIndexConfiguration(indexConfig, indexState));
            });
        }
      ).next(() => result);
    });
  }

  getNextCollectionGroupToUpdate(
    transaction: PersistenceTransaction
  ): PersistencePromise<string | null> {
    return this.getFieldIndexes(transaction).next(indexes => {
      if (indexes.length === 0) {
        return null;
      }
      indexes.sort(
        (l, r) => l.indexState.sequenceNumber - r.indexState.sequenceNumber
      );
      return indexes[0].collectionGroup;
    });
  }

  updateCollectionGroup(
    transaction: PersistenceTransaction,
    collectionGroup: string,
    offset: IndexOffset
  ): PersistencePromise<void> {
    const indexes = indexConfigurationStore(transaction);
    const states = indexStateStore(transaction);
    return this.getNextSequenceNumber(transaction).next(nextSequenceNumber =>
      indexes
        .loadAll(
          DbIndexConfiguration.collectionGroupIndex,
          IDBKeyRange.bound(collectionGroup, collectionGroup)
        )
        .next(configs =>
          PersistencePromise.forEach(configs, (config: DbIndexConfiguration) =>
            states.put(
              toDbIndexState(
                config.indexId!,
                this.user,
                nextSequenceNumber,
                offset
              )
            )
          )
        )
    );
  }

  updateIndexEntries(
    transaction: PersistenceTransaction,
    documents: DocumentMap
  ): PersistencePromise<void> {
    // TODO(indexing): Implement
    return PersistencePromise.resolve();
  }

  private getNextSequenceNumber(
    transaction: PersistenceTransaction
  ): PersistencePromise<number> {
    let nextSequenceNumber = 1;
    const states = indexStateStore(transaction);
    return states
      .iterate(
        {
          index: DbIndexState.sequenceNumberIndex,
          reverse: true,
          range: IDBKeyRange.upperBound([this.uid, Number.MAX_SAFE_INTEGER])
        },
        (_, state, controller) => {
          controller.done();
          nextSequenceNumber = state.sequenceNumber + 1;
        }
      )
      .next(() => nextSequenceNumber);
  }
}

/**
 * Helper to get a typed SimpleDbStore for the collectionParents
 * document store.
 */
function collectionParentsStore(
  txn: PersistenceTransaction
): SimpleDbStore<DbCollectionParentKey, DbCollectionParent> {
  return getStore<DbCollectionParentKey, DbCollectionParent>(
    txn,
    DbCollectionParent.store
  );
}

/**
 * Helper to get a typed SimpleDbStore for the index entry object store.
 */
function indexEntriesStore(
  txn: PersistenceTransaction
): SimpleDbStore<DbIndexEntryKey, DbIndexEntry> {
  return getStore<DbIndexEntryKey, DbIndexEntry>(txn, DbIndexEntry.store);
}

/**
 * Helper to get a typed SimpleDbStore for the index configuration object store.
 */
function indexConfigurationStore(
  txn: PersistenceTransaction
): SimpleDbStore<DbIndexConfigurationKey, DbIndexConfiguration> {
  return getStore<DbIndexConfigurationKey, DbIndexConfiguration>(
    txn,
    DbIndexConfiguration.store
  );
}

/**
 * Helper to get a typed SimpleDbStore for the index state object store.
 */
function indexStateStore(
  txn: PersistenceTransaction
): SimpleDbStore<DbIndexStateKey, DbIndexState> {
  return getStore<DbIndexStateKey, DbIndexState>(txn, DbIndexState.store);
}
