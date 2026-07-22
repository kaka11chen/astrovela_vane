// SPDX-FileCopyrightText: 2018-2025 Stichting DuckDB Foundation
// SPDX-FileCopyrightText: 2026 Vane contributors
// SPDX-License-Identifier: MIT
//
// Modified by Vane contributors.

#include "duckdb/main/relation/order_relation.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb/parser/query_node.hpp"
#include "duckdb/parser/query_node/select_node.hpp"
#include "duckdb/parser/expression/constant_expression.hpp"
#include "duckdb/parser/expression/star_expression.hpp"
#include "duckdb/main/config.hpp"
#include "duckdb/planner/binder.hpp"
#include "duckdb/planner/operator/logical_order.hpp"

namespace duckdb {

OrderRelation::OrderRelation(shared_ptr<Relation> child_p, vector<OrderByNode> orders)
    : Relation(child_p->context, RelationType::ORDER_RELATION), orders(std::move(orders)), child(std::move(child_p)) {
	D_ASSERT(child.get() != this);
	// bind the expressions
	TryBindRelation(columns);
}

unique_ptr<QueryNode> OrderRelation::GetQueryNode() {
	unique_ptr<QueryNode> result;
	if (RequiresSQLMultiSourceBinding(*child)) {
		result = child->GetQueryNode();
	} else {
		auto select = make_uniq<SelectNode>();
		select->from_table = child->GetTableRef();
		select->select_list.push_back(make_uniq<StarExpression>());
		result = std::move(select);
	}
	D_ASSERT(result->type == QueryNodeType::SELECT_NODE);
	auto &select = result->Cast<SelectNode>();
	auto order_node = make_uniq<OrderModifier>();
	for (idx_t i = 0; i < orders.size(); i++) {
		order_node->orders.emplace_back(orders[i].type, orders[i].null_order, orders[i].expression->Copy());
	}
	select.modifiers.push_back(std::move(order_node));
	return result;
}

BoundStatement OrderRelation::Bind(Binder &binder) {
	if (!RequiresDirectRelationBinding(*child)) {
		return Relation::Bind(binder);
	}
	auto select_node = make_uniq<SelectNode>();
	select_node->select_list.push_back(make_uniq<StarExpression>());
	auto order_node = make_uniq<OrderModifier>();
	for (auto &order : orders) {
		order_node->orders.emplace_back(order.type, order.null_order, order.expression->Copy());
	}
	select_node->modifiers.push_back(std::move(order_node));
	return BindSelectNodeOnChild(binder, *child, std::move(select_node));
}

BoundStatement OrderRelation::BindAsInput(Binder &binder) {
	auto child_ref = BindRelationInput(binder, *child);
	auto child_bound = binder.Bind(*child_ref);
	vector<unique_ptr<ParsedExpression>> visible_columns;
	ExpandRelationStar(binder, make_uniq<StarExpression>(), visible_columns);

	auto &config = DBConfig::GetConfig(binder.context);
	ExpressionBinder expression_binder(binder, binder.context);
	vector<BoundOrderByNode> bound_orders;
	for (auto &order : orders) {
		auto order_type = config.ResolveOrder(binder.context, order.type);
		auto null_order = config.ResolveNullOrder(binder.context, order_type, order.null_order);
		vector<unique_ptr<ParsedExpression>> order_expressions;
		ExpandRelationStar(binder, order.expression->Copy(), order_expressions);
		for (auto &expression : order_expressions) {
			if (expression->GetExpressionClass() == ExpressionClass::CONSTANT) {
				auto &constant = expression->Cast<ConstantExpression>();
				if (constant.value.type().IsIntegral()) {
					auto position = constant.value.GetValue<int64_t>();
					if (position <= 0 || static_cast<idx_t>(position) > visible_columns.size()) {
						throw BinderException("ORDER BY position %lld is not in select list", position);
					}
					expression = visible_columns[static_cast<idx_t>(position - 1)]->Copy();
				}
			}

			auto bound_expression = expression_binder.Bind(expression);
			ExpressionBinder::PushCollation(binder.context, bound_expression, bound_expression->return_type);
			bound_orders.emplace_back(order_type, null_order, std::move(bound_expression));
		}
	}
	for (auto &order : bound_orders) {
		PlanRelationSubqueries(binder, order.expression, child_bound.plan);
	}
	auto order = make_uniq<LogicalOrder>(std::move(bound_orders));
	order->AddChild(std::move(child_bound.plan));
	child_bound.plan = std::move(order);
	return child_bound;
}

bool OrderRelation::CanSerializeToQueryNode() {
	for (auto &order : orders) {
		if (!CanSerializeExpressionOnChild(*child, *order.expression)) {
			return false;
		}
	}
	return true;
}

string OrderRelation::GetAlias() {
	return child->GetAlias();
}

const vector<ColumnDefinition> &OrderRelation::Columns() {
	return columns;
}

string OrderRelation::ToString(idx_t depth) {
	string str = RenderWhitespace(depth) + "Order [";
	for (idx_t i = 0; i < orders.size(); i++) {
		if (i != 0) {
			str += ", ";
		}
		str += orders[i].expression->ToString() + (orders[i].type == OrderType::ASCENDING ? " ASC" : " DESC");
	}
	str += "]\n";
	return str + child->ToString(depth + 1);
}

} // namespace duckdb
