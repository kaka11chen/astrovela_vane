// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT

#include "duckdb/main/relation/local_exchange_relation.hpp"
#include "duckdb/common/exception.hpp"
#include "duckdb/execution/distributed/common_types.hpp"
#include "duckdb/execution/operator/exchange/repartition.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_local_exchange.hpp"

namespace duckdb {

LocalExchangeRelation::LocalExchangeRelation(shared_ptr<Relation> child_p, idx_t num_partitions_p)
    : Relation(child_p->context, RelationType::LOCAL_EXCHANGE_RELATION), num_partitions(num_partitions_p),
      child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	TryBindRelation(columns);
}

unique_ptr<QueryNode> LocalExchangeRelation::GetQueryNode() {
	throw NotImplementedException(
	    "A local-exchange relation has no SQL query-node representation; converting it would discard the exchange");
}

string LocalExchangeRelation::GetQuery() {
	return string();
}

string LocalExchangeRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &LocalExchangeRelation::Columns() {
	return columns;
}

BoundStatement LocalExchangeRelation::Bind(Binder &binder) {
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	return BindSelectNodeOnChild(binder, *this, std::move(select_node));
}

BoundStatement LocalExchangeRelation::BindAsInput(Binder &binder) {
	auto child_ref = BindRelationInput(binder, *child);
	auto child_bound = binder.Bind(*child_ref);

	size_t num_partitions_sz = static_cast<size_t>(num_partitions);
	auto spec = RepartitionSpec::create_random(num_partitions_sz);

	auto local_exchange = make_uniq<LogicalLocalExchange>(std::move(spec));
	local_exchange->children.push_back(std::move(child_bound.plan));
	child_bound.plan = std::move(local_exchange);
	return child_bound;
}

string LocalExchangeRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "LocalExchange";
	if (num_partitions) {
		str += " [" + std::to_string(num_partitions) + "]";
	}
	str += "\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
