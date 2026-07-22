// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/distinct_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/planner/operator/logical_distinct.hpp"

namespace duckdb {

DistinctRelation::DistinctRelation(shared_ptr<Relation> child_p)
    : Relation(child_p->context, RelationType::DISTINCT_RELATION), child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	vector<ColumnDefinition> dummy_columns;
	TryBindRelation(dummy_columns);
}

unique_ptr<QueryNode> DistinctRelation::GetQueryNode() {
	auto child_node = child->GetQueryNode();
	child_node->AddDistinct();
	return child_node;
}

BoundStatement DistinctRelation::Bind(Binder &binder) {
	if (!RequiresDirectRelationBinding(*child)) {
		return Relation::Bind(binder);
	}
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	select_node->AddDistinct();
	return BindSelectNodeOnChild(binder, *child, std::move(select_node));
}

BoundStatement DistinctRelation::BindAsInput(Binder &binder) {
	auto child_ref = BindRelationInput(binder, *child);
	auto child_bound = binder.Bind(*child_ref);
	auto bindings = child_bound.plan->GetColumnBindings();
	D_ASSERT(bindings.size() == child_bound.names.size());
	D_ASSERT(bindings.size() == child_bound.types.size());

	vector<unique_ptr<Expression>> targets;
	targets.reserve(bindings.size());
	for (idx_t i = 0; i < bindings.size(); i++) {
		unique_ptr<Expression> target =
		    make_uniq<BoundColumnRefExpression>(child_bound.names[i], child_bound.types[i], bindings[i]);
		ExpressionBinder::PushCollation(binder.context, target, target->return_type);
		targets.push_back(std::move(target));
	}
	auto distinct = make_uniq<LogicalDistinct>(std::move(targets), DistinctType::DISTINCT);
	distinct->AddChild(std::move(child_bound.plan));
	child_bound.plan = std::move(distinct);
	return child_bound;
}

string DistinctRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &DistinctRelation::Columns() {
	return child->Columns();
}

string DistinctRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Distinct\n";
	return str + child->ToString(depth + 1);
	;
}

} // namespace duckdb
