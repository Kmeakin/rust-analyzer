//! Transforms `ast::Expr` into an equivalent `hir_def::expr::Expr`
//! representation.

use std::mem;

use base_db::CrateId;
use either::Either;
use hir_expand::{
    ast_id_map::AstIdMap,
    name::{name, AsName, Name},
    AstId, ExpandError, InFile,
};
use intern::Interned;
use la_arena::{Arena, ArenaMap};
use profile::Count;
use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use syntax::{
    ast::{
        self, ArrayExprKind, AstChildren, BlockExpr, HasArgList, HasAttrs, HasLoopBody, HasName,
        SlicePatComponents,
    },
    AstNode, AstPtr, SyntaxNodePtr,
};
use triomphe::Arc;

use crate::{
    body::{Body, BodyDiagnostic, BodySourceMap, ExprPtr, ExprSource, LabelPtr, PatPtr},
    data::adt::StructKind,
    db::DefDatabase,
    expander::Expander,
    hir::{
        dummy_expr_id, Array, Binding, BindingAnnotation, BindingId, BindingProblems, CaptureBy,
        ClosureKind, Expr, ExprId, ExprRange, Label, LabelId, Literal, LiteralOrConst, MatchArm,
        Movability, Pat, PatId, PatRange, RecordFieldPat, RecordLitField, Statement, StmtRange,
    },
    item_scope::BuiltinShadowMode,
    lang_item::LangItem,
    lower::LowerCtx,
    nameres::{DefMap, MacroSubNs},
    path::{GenericArgs, Path},
    type_ref::{Mutability, Rawness, TypeRef},
    AdtId, BlockId, BlockLoc, DefWithBodyId, ModuleDefId, UnresolvedMacro,
};

pub(super) fn lower(
    db: &dyn DefDatabase,
    owner: DefWithBodyId,
    expander: Expander,
    params: Option<(ast::ParamList, impl Iterator<Item = bool>)>,
    body: Option<ast::Expr>,
    krate: CrateId,
    is_async_fn: bool,
) -> (Body, BodySourceMap) {
    ExprCollector {
        db,
        owner,
        krate,
        def_map: expander.module.def_map(db),
        source_map: BodySourceMap::default(),
        ast_id_map: db.ast_id_map(expander.current_file_id),
        body: Body {
            exprs: Arena::default(),
            pats: Arena::default(),
            stmts: Arena::default(),
            record_field_pats: Arena::default(),
            bindings: Arena::default(),
            labels: Arena::default(),
            params: PatRange::empty(),
            body_expr: dummy_expr_id(),
            block_scopes: Vec::new(),
            _c: Count::new(),
        },
        expander,
        current_try_block_label: None,
        is_lowering_assignee_expr: false,
        is_lowering_generator: false,
        label_ribs: Vec::new(),
        current_binding_owner: None,
        binding_owners: ArenaMap::default(),
    }
    .collect(params, body, is_async_fn)
}

struct ExprCollector<'a> {
    db: &'a dyn DefDatabase,
    expander: Expander,
    owner: DefWithBodyId,
    def_map: Arc<DefMap>,
    ast_id_map: Arc<AstIdMap>,
    krate: CrateId,
    body: Body,
    source_map: BodySourceMap,

    is_lowering_assignee_expr: bool,
    is_lowering_generator: bool,

    current_try_block_label: Option<LabelId>,
    // points to the expression that a try expression will target (replaces current_try_block_label)
    // catch_scope: Option<ExprId>,
    // points to the expression that an unlabeled control flow will target
    // loop_scope: Option<ExprId>,
    // needed to diagnose non label control flow in while conditions
    // is_in_loop_condition: bool,

    // resolution
    label_ribs: Vec<LabelRib>,
    current_binding_owner: Option<ExprSource>,
    binding_owners: ArenaMap<BindingId, ExprSource>,
}

#[derive(Clone, Debug)]
struct LabelRib {
    kind: RibKind,
    // Once we handle macro hygiene this will need to be a map
    label: Option<(Name, LabelId)>,
}

impl LabelRib {
    fn new(kind: RibKind) -> Self {
        LabelRib { kind, label: None }
    }
    fn new_normal(label: (Name, LabelId)) -> Self {
        LabelRib { kind: RibKind::Normal, label: Some(label) }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RibKind {
    Normal,
    Closure,
    Constant,
}

impl RibKind {
    /// This rib forbids referring to labels defined in upwards ribs.
    fn is_label_barrier(self) -> bool {
        match self {
            RibKind::Normal => false,
            RibKind::Closure | RibKind::Constant => true,
        }
    }
}

#[derive(Debug, Default)]
struct BindingList {
    map: FxHashMap<Name, BindingId>,
    is_used: FxHashMap<BindingId, bool>,
    reject_new: bool,
}

impl BindingList {
    fn find(
        &mut self,
        ec: &mut ExprCollector<'_>,
        name: Name,
        mode: BindingAnnotation,
    ) -> BindingId {
        let id = *self.map.entry(name).or_insert_with_key(|n| ec.alloc_binding(n.clone(), mode));
        if ec.body.bindings[id].mode != mode {
            ec.body.bindings[id].problems = Some(BindingProblems::BoundInconsistently);
        }
        self.check_is_used(ec, id);
        id
    }

    fn check_is_used(&mut self, ec: &mut ExprCollector<'_>, id: BindingId) {
        match self.is_used.get(&id) {
            None => {
                if self.reject_new {
                    ec.body.bindings[id].problems = Some(BindingProblems::NotBoundAcrossAll);
                }
            }
            Some(true) => {
                ec.body.bindings[id].problems = Some(BindingProblems::BoundMoreThanOnce);
            }
            Some(false) => {}
        }
        self.is_used.insert(id, true);
    }
}

impl ExprCollector<'_> {
    fn collect(
        mut self,
        param_list: Option<(ast::ParamList, impl Iterator<Item = bool>)>,
        body: Option<ast::Expr>,
        is_async_fn: bool,
    ) -> (Body, BodySourceMap) {
        if let Some((param_list, mut attr_enabled)) = param_list {
            let mut params = Vec::new();
            if let Some(self_param) =
                param_list.self_param().filter(|_| attr_enabled.next().unwrap_or(false))
            {
                let ptr = AstPtr::new(&self_param);
                let binding_id = self.alloc_binding(
                    name![self],
                    BindingAnnotation::new(
                        self_param.mut_token().is_some() && self_param.amp_token().is_none(),
                        false,
                    ),
                );
                let param_pat = (
                    Pat::Bind { id: binding_id, subpat: None },
                    PatOrigin::User(self.expander.to_source(Either::Right(ptr))),
                    Some(binding_id),
                );
                params.push(param_pat);
            }

            for (param, _) in param_list.params().zip(attr_enabled).filter(|(_, enabled)| *enabled)
            {
                let param_pat = self.lower_pat_top(param.pat());
                params.push(param_pat);
            }
            self.body.params = self.alloc_pats(params);
        };
        self.body.body_expr = self.with_label_rib(RibKind::Closure, |this| {
            if is_async_fn {
                match body {
                    Some(e) => {
                        let expr = this.collect_expr(e);
                        this.alloc_expr_desugared(Expr::Async {
                            id: None,
                            statements: StmtRange::empty(),
                            tail: Some(expr),
                        })
                    }
                    None => this.missing_expr(),
                }
            } else {
                this.collect_expr_opt(body)
            }
        });

        for (id, owner) in self.binding_owners {
            self.body.bindings[id].owner = Some(self.source_map.expr_map[&owner]);
        }

        (self.body, self.source_map)
    }

    fn ctx(&self) -> LowerCtx<'_> {
        self.expander.ctx(self.db)
    }

    fn collect_expr(&mut self, expr: ast::Expr) -> ExprId {
        let (expr, origin) = self.lower_expr(expr);
        self.alloc_expr(expr, origin)
    }

    fn lower_expr(&mut self, expr: ast::Expr) -> (Expr, ExprOrigin) {
        self.maybe_lower_expr(expr).unwrap_or_else(|| (Expr::Missing, ExprOrigin::None))
    }

    /// Returns `None` if and only if the expression is `#[cfg]`d out.
    fn maybe_collect_expr(&mut self, expr: ast::Expr) -> Option<ExprId> {
        let (expr, origin) = self.maybe_lower_expr(expr)?;
        Some(self.alloc_expr(expr, origin))
    }

    fn maybe_lower_expr(&mut self, expr: ast::Expr) -> Option<(Expr, ExprOrigin)> {
        let syntax_ptr = AstPtr::new(&expr);
        self.check_cfg(&expr)?;

        // FIXME: Move some of these arms out into separate methods for clarity
        let expression = match expr {
            ast::Expr::IfExpr(e) => {
                let then_branch = self.collect_block_opt(e.then_branch());

                let else_branch = e.else_branch().map(|b| match b {
                    ast::ElseBranch::Block(it) => self.collect_block(it),
                    ast::ElseBranch::IfExpr(elif) => {
                        let expr: ast::Expr = ast::Expr::cast(elif.syntax().clone()).unwrap();
                        self.collect_expr(expr)
                    }
                });

                let condition = self.collect_expr_opt(e.condition());

                Expr::If { condition, then_branch, else_branch }
            }
            ast::Expr::LetExpr(e) => {
                let pat = self.collect_pat_top(e.pat());
                let expr = self.collect_expr_opt(e.expr());
                Expr::Let { pat, expr }
            }
            ast::Expr::BlockExpr(e) => return Some(self.lower_block_expr(e, syntax_ptr)),
            ast::Expr::LoopExpr(e) => {
                let label = e.label().map(|label| self.collect_label(label));
                let body = self.collect_labelled_block_opt(label, e.loop_body());
                Expr::Loop { body, label }
            }
            ast::Expr::WhileExpr(e) => {
                let label = e.label().map(|label| self.collect_label(label));
                let body = self.collect_labelled_block_opt(label, e.loop_body());
                let condition = self.collect_expr_opt(e.condition());

                Expr::While { condition, body, label }
            }
            ast::Expr::ForExpr(e) => return Some(self.collect_for_loop(syntax_ptr, e)),
            ast::Expr::CallExpr(e) => {
                let is_rustc_box = {
                    let attrs = e.attrs();
                    attrs.filter_map(|x| x.as_simple_atom()).any(|x| x == "rustc_box")
                };
                if is_rustc_box {
                    let expr = self.collect_expr_opt(e.arg_list().and_then(|x| x.args().next()));
                    Expr::Box { expr }
                } else {
                    let callee = self.collect_expr_opt(e.expr());
                    let args = if let Some(arg_list) = e.arg_list() {
                        arg_list.args().filter_map(|e| self.maybe_lower_expr(e)).collect()
                    } else {
                        Vec::default()
                    };
                    let args = self.alloc_exprs(args);
                    Expr::Call { callee, args, is_assignee_expr: self.is_lowering_assignee_expr }
                }
            }
            ast::Expr::MethodCallExpr(e) => {
                let receiver = self.collect_expr_opt(e.receiver());
                let args = if let Some(arg_list) = e.arg_list() {
                    arg_list.args().filter_map(|e| self.maybe_lower_expr(e)).collect()
                } else {
                    Vec::default()
                };
                let args = self.alloc_exprs(args);
                let method_name = e.name_ref().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                let generic_args = e
                    .generic_arg_list()
                    .and_then(|it| GenericArgs::from_ast(&self.ctx(), it))
                    .map(Box::new);
                Expr::MethodCall { receiver, method_name, args, generic_args }
            }
            ast::Expr::MatchExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let arms = if let Some(match_arm_list) = e.match_arm_list() {
                    match_arm_list
                        .arms()
                        .filter_map(|arm| {
                            self.check_cfg(&arm).map(|()| MatchArm {
                                pat: self.collect_pat_top(arm.pat()),
                                expr: self.collect_expr_opt(arm.expr()),
                                guard: arm
                                    .guard()
                                    .map(|guard| self.collect_expr_opt(guard.condition())),
                            })
                        })
                        .collect()
                } else {
                    Box::default()
                };
                Expr::Match { expr, arms }
            }
            ast::Expr::PathExpr(e) => e
                .path()
                .and_then(|path| self.expander.parse_path(self.db, path))
                .map(Expr::Path)
                .unwrap_or(Expr::Missing),
            ast::Expr::ContinueExpr(e) => {
                let label = self.resolve_label(e.lifetime()).unwrap_or_else(|e| {
                    self.source_map.diagnostics.push(e);
                    None
                });
                Expr::Continue { label }
            }
            ast::Expr::BreakExpr(e) => {
                let label = self.resolve_label(e.lifetime()).unwrap_or_else(|e| {
                    self.source_map.diagnostics.push(e);
                    None
                });
                let expr = e.expr().map(|e| self.collect_expr(e));
                Expr::Break { expr, label }
            }
            ast::Expr::ParenExpr(e) => {
                let (expr, origin) = self.lower_expr_opt(e.expr());
                // make the paren expr point to the inner expression as well
                let src = self.expander.to_source(syntax_ptr);
                return Some((expr, ExprOrigin::Macro(src, Box::new(origin))));
            }
            ast::Expr::ReturnExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                Expr::Return { expr }
            }
            ast::Expr::YieldExpr(e) => {
                self.is_lowering_generator = true;
                let expr = e.expr().map(|e| self.collect_expr(e));
                Expr::Yield { expr }
            }
            ast::Expr::YeetExpr(e) => {
                let expr = e.expr().map(|e| self.collect_expr(e));
                Expr::Yeet { expr }
            }
            ast::Expr::RecordExpr(e) => {
                let path =
                    e.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let is_assignee_expr = self.is_lowering_assignee_expr;

                match e.record_expr_field_list() {
                    Some(nfl) => {
                        let fields = nfl
                            .fields()
                            .filter_map(|field| {
                                self.check_cfg(&field)?;

                                let name = field.field_name()?.as_name();

                                let expr = match field.expr() {
                                    Some(e) => self.collect_expr(e),
                                    None => self.missing_expr(),
                                };
                                let src = self.expander.to_source(AstPtr::new(&field));
                                self.source_map.field_map.insert(src.clone(), expr);
                                self.source_map.field_map_back.insert(expr, src);
                                Some(RecordLitField { name, expr })
                            })
                            .collect();
                        let spread = nfl.spread().map(|s| self.collect_expr(s));
                        let ellipsis = nfl.dotdot_token().is_some();
                        Expr::RecordLit { path, fields, spread, ellipsis, is_assignee_expr }
                    }
                    None => Expr::RecordLit {
                        path,
                        fields: Box::default(),
                        spread: None,
                        ellipsis: false,
                        is_assignee_expr,
                    },
                }
            }
            ast::Expr::FieldExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let name = match e.field_access() {
                    Some(kind) => kind.as_name(),
                    _ => Name::missing(),
                };
                Expr::Field { expr, name }
            }
            ast::Expr::AwaitExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                Expr::Await { expr }
            }
            ast::Expr::TryExpr(e) => return Some(self.collect_try_operator(syntax_ptr, e)),
            ast::Expr::CastExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let type_ref = Interned::new(TypeRef::from_ast_opt(&self.ctx(), e.ty()));
                Expr::Cast { expr, type_ref }
            }
            ast::Expr::RefExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                let raw_tok = e.raw_token().is_some();
                let mutability = if raw_tok {
                    if e.mut_token().is_some() {
                        Mutability::Mut
                    } else if e.const_token().is_some() {
                        Mutability::Shared
                    } else {
                        unreachable!("parser only remaps to raw_token() if matching mutability token follows")
                    }
                } else {
                    Mutability::from_mutable(e.mut_token().is_some())
                };
                let rawness = Rawness::from_raw(raw_tok);
                Expr::Ref { expr, rawness, mutability }
            }
            ast::Expr::PrefixExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                match e.op_kind() {
                    Some(op) => Expr::UnaryOp { expr, op },
                    None => Expr::Missing,
                }
            }
            ast::Expr::ClosureExpr(e) => self.with_label_rib(RibKind::Closure, |this| {
                let prev_binding_owner = this.current_binding_owner.take();
                this.current_binding_owner = Some(this.expander.to_source(syntax_ptr.clone()));
                let mut args = Vec::new();
                let mut arg_types = Vec::new();
                if let Some(pl) = e.param_list() {
                    for param in pl.params() {
                        let pat = this.lower_pat_top(param.pat());
                        let type_ref =
                            param.ty().map(|it| Interned::new(TypeRef::from_ast(&this.ctx(), it)));
                        args.push(pat);
                        arg_types.push(type_ref);
                    }
                }
                let args = this.alloc_pats(args);
                let ret_type = e
                    .ret_type()
                    .and_then(|r| r.ty())
                    .map(|it| Interned::new(TypeRef::from_ast(&this.ctx(), it)));

                let prev_is_lowering_generator = mem::take(&mut this.is_lowering_generator);
                let prev_try_block_label = this.current_try_block_label.take();

                let body = this.collect_expr_opt(e.body());

                let closure_kind = if this.is_lowering_generator {
                    let movability = if e.static_token().is_some() {
                        Movability::Static
                    } else {
                        Movability::Movable
                    };
                    ClosureKind::Generator(movability)
                } else if e.async_token().is_some() {
                    ClosureKind::Async
                } else {
                    ClosureKind::Closure
                };
                let capture_by =
                    if e.move_token().is_some() { CaptureBy::Value } else { CaptureBy::Ref };
                this.is_lowering_generator = prev_is_lowering_generator;
                this.current_binding_owner = prev_binding_owner;
                this.current_try_block_label = prev_try_block_label;
                Expr::Closure {
                    args,
                    arg_types: arg_types.into(),
                    ret_type,
                    body,
                    closure_kind,
                    capture_by,
                }
            }),
            ast::Expr::BinExpr(e) => {
                let op = e.op_kind();
                if let Some(ast::BinaryOp::Assignment { op: None }) = op {
                    self.is_lowering_assignee_expr = true;
                }
                let lhs = self.collect_expr_opt(e.lhs());
                self.is_lowering_assignee_expr = false;
                let rhs = self.collect_expr_opt(e.rhs());
                Expr::BinaryOp { lhs, rhs, op }
            }
            ast::Expr::TupleExpr(e) => {
                let exprs = e.fields().map(|expr| self.lower_expr(expr));
                // if there is a leading comma, the user is most likely to type out a leading expression
                // so we insert a missing expression at the beginning for IDE features
                let first = if comma_follows_token(e.l_paren_token()) {
                    Some((Expr::Missing, ExprOrigin::None))
                } else {
                    None
                };

                let exprs: Vec<_> = first.into_iter().chain(exprs).collect();
                let exprs = self.alloc_exprs(exprs);

                Expr::Tuple { exprs, is_assignee_expr: self.is_lowering_assignee_expr }
            }
            ast::Expr::BoxExpr(e) => {
                let expr = self.collect_expr_opt(e.expr());
                Expr::Box { expr }
            }

            ast::Expr::ArrayExpr(e) => {
                let kind = e.kind();

                match kind {
                    ArrayExprKind::ElementList(e) => {
                        let elements: Vec<_> = e.map(|expr| self.lower_expr(expr)).collect();
                        let elements = self.alloc_exprs(elements);
                        Expr::Array(Array::ElementList {
                            elements,
                            is_assignee_expr: self.is_lowering_assignee_expr,
                        })
                    }
                    ArrayExprKind::Repeat { initializer, repeat } => {
                        let initializer = self.collect_expr_opt(initializer);
                        let repeat = self.with_label_rib(RibKind::Constant, |this| {
                            if let Some(repeat) = repeat {
                                let syntax_ptr = AstPtr::new(&repeat);
                                let prev_binding_owner = this.current_binding_owner.take();
                                this.current_binding_owner =
                                    Some(this.expander.to_source(syntax_ptr));
                                let result = this.collect_expr(repeat);
                                this.current_binding_owner = prev_binding_owner;
                                result
                            } else {
                                this.missing_expr()
                            }
                        });
                        Expr::Array(Array::Repeat { initializer, repeat })
                    }
                }
            }

            ast::Expr::Literal(e) => Expr::Literal(e.kind().into()),
            ast::Expr::IndexExpr(e) => {
                let base = self.collect_expr_opt(e.base());
                let index = self.collect_expr_opt(e.index());
                Expr::Index { base, index }
            }
            ast::Expr::RangeExpr(e) => {
                let lhs = e.start().map(|lhs| self.collect_expr(lhs));
                let rhs = e.end().map(|rhs| self.collect_expr(rhs));
                match e.op_kind() {
                    Some(range_type) => Expr::Range { lhs, rhs, range_type },
                    None => Expr::Missing,
                }
            }
            ast::Expr::MacroExpr(e) => {
                let e = e.macro_call()?;
                let macro_ptr = AstPtr::new(&e);
                let it = self.collect_macro_call(e, macro_ptr, true, |this, expansion| {
                    expansion.map(|it| this.lower_expr(it))
                });
                match it {
                    Some((expr, origin)) => {
                        // Make the macro-call point to its expanded expression so we can query
                        // semantics on syntax pointers to the macro
                        let src = self.expander.to_source(syntax_ptr);
                        return Some((expr, ExprOrigin::Macro(src, Box::new(origin))));
                    }
                    None => Expr::Missing,
                }
            }
            ast::Expr::UnderscoreExpr(_) => Expr::Underscore,
        };
        Some((expression, ExprOrigin::User(self.expander.to_source(syntax_ptr))))
    }

    fn lower_block_expr(
        &mut self,
        block: ast::BlockExpr,
        syntax_ptr: AstPtr<ast::Expr>,
    ) -> (Expr, ExprOrigin) {
        match block.modifier() {
            Some(ast::BlockModifier::Try(_)) => self.desugar_try_block(block),
            Some(ast::BlockModifier::Unsafe(_)) => self
                .lower_block_(block, |id, statements, tail| Expr::Unsafe { id, statements, tail }),
            Some(ast::BlockModifier::Label(label)) => {
                let label = self.collect_label(label);
                self.with_labeled_rib(label, |this| {
                    this.lower_block_(block, |id, statements, tail| Expr::Block {
                        id,
                        statements,
                        tail,
                        label: Some(label),
                    })
                })
            }
            Some(ast::BlockModifier::Async(_)) => self.with_label_rib(RibKind::Closure, |this| {
                this.lower_block_(block, |id, statements, tail| Expr::Async {
                    id,
                    statements,
                    tail,
                })
            }),
            Some(ast::BlockModifier::Const(_)) => self.with_label_rib(RibKind::Constant, |this| {
                let origin = ExprOrigin::User(this.expander.to_source(syntax_ptr.clone()));
                let prev_binding_owner = this.current_binding_owner.take();
                this.current_binding_owner = Some(this.expander.to_source(syntax_ptr));
                let inner_expr = this.collect_block(block);
                let const_id = this.db.intern_anonymous_const((this.owner, inner_expr));
                let expr = Expr::Const(const_id);
                this.current_binding_owner = prev_binding_owner;
                (expr, origin)
            }),
            None => self.lower_block(block),
        }
    }

    /// Desugar `try { <stmts>; <expr> }` into `'<new_label>: { <stmts>; ::std::ops::Try::from_output(<expr>) }`,
    /// `try { <stmts>; }` into `'<new_label>: { <stmts>; ::std::ops::Try::from_output(()) }`
    /// and save the `<new_label>` to use it as a break target for desugaring of the `?` operator.
    fn desugar_try_block(&mut self, e: BlockExpr) -> (Expr, ExprOrigin) {
        let Some(try_from_output) = LangItem::TryTraitFromOutput.path(self.db, self.krate) else {
            return self.lower_block(e);
        };
        let label = self.alloc_label_desugared(Label { name: Name::generate_new_name() });
        let old_label = self.current_try_block_label.replace(label);

        let (btail, (mut expr, origin)) = self.with_labeled_rib(label, |this| {
            let mut btail = None;
            let block = this.lower_block_(e, |id, statements, tail| {
                btail = tail;
                Expr::Block { id, statements, tail, label: Some(label) }
            });
            (btail, block)
        });

        let callee = self.alloc_expr_desugared(Expr::Path(try_from_output));
        let next_tail = match btail {
            Some(tail) => self.alloc_expr_desugared(Expr::Call {
                callee,
                args: ExprRange::singleton(tail),
                is_assignee_expr: false,
            }),
            None => {
                let unit = self.alloc_expr_desugared(Expr::Tuple {
                    exprs: ExprRange::empty(),
                    is_assignee_expr: false,
                });
                self.alloc_expr_desugared(Expr::Call {
                    callee,
                    args: ExprRange::singleton(unit),
                    is_assignee_expr: false,
                })
            }
        };
        let Expr::Block { tail, .. } = &mut expr else {
            unreachable!("block was lowered to non-block");
        };
        *tail = Some(next_tail);
        self.current_try_block_label = old_label;
        (expr, origin)
    }

    /// Desugar `ast::ForExpr` from: `[opt_ident]: for <pat> in <head> <body>` into:
    /// ```ignore (pseudo-rust)
    /// match IntoIterator::into_iter(<head>) {
    ///     mut iter => {
    ///         [opt_ident]: loop {
    ///             match Iterator::next(&mut iter) {
    ///                 None => break,
    ///                 Some(<pat>) => <body>,
    ///             };
    ///         }
    ///     }
    /// }
    /// ```
    fn collect_for_loop(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
        e: ast::ForExpr,
    ) -> (Expr, ExprOrigin) {
        let origin = ExprOrigin::User(self.expander.to_source(syntax_ptr));

        let (into_iter_fn, iter_next_fn, option_some, option_none) = 'if_chain: {
            if let Some(into_iter_fn) = LangItem::IntoIterIntoIter.path(self.db, self.krate) {
                if let Some(iter_next_fn) = LangItem::IteratorNext.path(self.db, self.krate) {
                    if let Some(option_some) = LangItem::OptionSome.path(self.db, self.krate) {
                        if let Some(option_none) = LangItem::OptionNone.path(self.db, self.krate) {
                            break 'if_chain (into_iter_fn, iter_next_fn, option_some, option_none);
                        }
                    }
                }
            }
            // Some of the needed lang items are missing, so we can't desugar
            return (Expr::Missing, origin);
        };
        let head = self.collect_expr_opt(e.iterable());
        let into_iter_fn_expr = self.alloc_expr(Expr::Path(into_iter_fn), origin.clone());
        let iterator = self.alloc_expr(
            Expr::Call {
                callee: into_iter_fn_expr,
                args: ExprRange::singleton(head),
                is_assignee_expr: false,
            },
            origin.clone(),
        );
        let none_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::Path(Box::new(option_none))),
            guard: None,
            expr: self.alloc_expr(Expr::Break { expr: None, label: None }, origin.clone()),
        };
        let some_pat = Pat::TupleStruct {
            path: Some(Box::new(option_some)),
            args: PatRange::singleton(self.collect_pat_top(e.pat())),
            ellipsis: None,
        };
        let label = e.label().map(|label| self.collect_label(label));
        let some_arm = MatchArm {
            pat: self.alloc_pat_desugared(some_pat),
            guard: None,
            expr: self.with_opt_labeled_rib(label, |this| {
                this.collect_expr_opt(e.loop_body().map(|x| x.into()))
            }),
        };
        let iter_name = Name::generate_new_name();
        let iter_binding = self.alloc_binding(iter_name.clone(), BindingAnnotation::Mutable);
        let iter_expr = self.alloc_expr(Expr::Path(Path::from(iter_name)), origin.clone());
        let iter_expr_mut = self.alloc_expr(
            Expr::Ref { expr: iter_expr, rawness: Rawness::Ref, mutability: Mutability::Mut },
            origin.clone(),
        );
        let iter_next_fn_expr = self.alloc_expr(Expr::Path(iter_next_fn), origin.clone());
        let iter_next_expr = self.alloc_expr(
            Expr::Call {
                callee: iter_next_fn_expr,
                args: ExprRange::singleton(iter_expr_mut),
                is_assignee_expr: false,
            },
            origin.clone(),
        );
        let loop_inner = self.alloc_expr(
            Expr::Match { expr: iter_next_expr, arms: Box::new([none_arm, some_arm]) },
            origin.clone(),
        );
        let loop_outer = self.alloc_expr(Expr::Loop { body: loop_inner, label }, origin.clone());
        let iter_pat = self.alloc_pat_desugared(Pat::Bind { id: iter_binding, subpat: None });
        (
            Expr::Match {
                expr: iterator,
                arms: Box::new([MatchArm { pat: iter_pat, guard: None, expr: loop_outer }]),
            },
            origin,
        )
    }

    /// Desugar `ast::TryExpr` from: `<expr>?` into:
    /// ```ignore (pseudo-rust)
    /// match Try::branch(<expr>) {
    ///     ControlFlow::Continue(val) => val,
    ///     ControlFlow::Break(residual) =>
    ///         // If there is an enclosing `try {...}`:
    ///         break 'catch_target Try::from_residual(residual),
    ///         // Otherwise:
    ///         return Try::from_residual(residual),
    /// }
    /// ```
    fn collect_try_operator(
        &mut self,
        syntax_ptr: AstPtr<ast::Expr>,
        e: ast::TryExpr,
    ) -> (Expr, ExprOrigin) {
        let origin = ExprOrigin::User(self.expander.to_source(syntax_ptr));

        let (try_branch, cf_continue, cf_break, try_from_residual) = 'if_chain: {
            if let Some(try_branch) = LangItem::TryTraitBranch.path(self.db, self.krate) {
                if let Some(cf_continue) = LangItem::ControlFlowContinue.path(self.db, self.krate) {
                    if let Some(cf_break) = LangItem::ControlFlowBreak.path(self.db, self.krate) {
                        if let Some(try_from_residual) =
                            LangItem::TryTraitFromResidual.path(self.db, self.krate)
                        {
                            break 'if_chain (try_branch, cf_continue, cf_break, try_from_residual);
                        }
                    }
                }
            }
            // Some of the needed lang items are missing, so we can't desugar
            return (Expr::Missing, origin);
        };
        let operand = self.collect_expr_opt(e.expr());
        let try_branch = self.alloc_expr(Expr::Path(try_branch), origin.clone());
        let expr = self.alloc_expr(
            Expr::Call {
                callee: try_branch,
                args: ExprRange::singleton(operand),
                is_assignee_expr: false,
            },
            origin.clone(),
        );
        let continue_name = Name::generate_new_name();
        let continue_binding =
            self.alloc_binding(continue_name.clone(), BindingAnnotation::Unannotated);
        let continue_bpat = self.alloc_pat(
            Pat::Bind { id: continue_binding, subpat: None },
            PatOrigin::None,
            Some(continue_binding),
        );
        let continue_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::TupleStruct {
                path: Some(Box::new(cf_continue)),
                args: PatRange::singleton(continue_bpat),
                ellipsis: None,
            }),
            guard: None,
            expr: self.alloc_expr(Expr::Path(Path::from(continue_name)), origin.clone()),
        };
        let break_name = Name::generate_new_name();
        let break_binding = self.alloc_binding(break_name.clone(), BindingAnnotation::Unannotated);
        let break_bpat = self.alloc_pat(
            Pat::Bind { id: break_binding, subpat: None },
            PatOrigin::None,
            Some(break_binding),
        );
        let break_arm = MatchArm {
            pat: self.alloc_pat_desugared(Pat::TupleStruct {
                path: Some(Box::new(cf_break)),
                args: PatRange::singleton(break_bpat),
                ellipsis: None,
            }),
            guard: None,
            expr: {
                let x = self.alloc_expr(Expr::Path(Path::from(break_name)), origin.clone());
                let callee = self.alloc_expr(Expr::Path(try_from_residual), origin.clone());
                let result = self.alloc_expr(
                    Expr::Call { callee, args: ExprRange::singleton(x), is_assignee_expr: false },
                    origin.clone(),
                );
                self.alloc_expr(
                    match self.current_try_block_label {
                        Some(label) => Expr::Break { expr: Some(result), label: Some(label) },
                        None => Expr::Return { expr: Some(result) },
                    },
                    origin.clone(),
                )
            },
        };
        let arms = Box::new([continue_arm, break_arm]);
        (Expr::Match { expr, arms }, origin)
    }

    fn collect_macro_call<F, T, U>(
        &mut self,
        mcall: ast::MacroCall,
        syntax_ptr: AstPtr<ast::MacroCall>,
        record_diagnostics: bool,
        collector: F,
    ) -> U
    where
        F: FnOnce(&mut Self, Option<T>) -> U,
        T: ast::AstNode,
    {
        // File containing the macro call. Expansion errors will be attached here.
        let outer_file = self.expander.current_file_id;

        let macro_call_ptr = self.expander.to_source(AstPtr::new(&mcall));
        let module = self.expander.module.local_id;
        let res = self.expander.enter_expand(self.db, mcall, |path| {
            self.def_map
                .resolve_path(
                    self.db,
                    module,
                    &path,
                    crate::item_scope::BuiltinShadowMode::Other,
                    Some(MacroSubNs::Bang),
                )
                .0
                .take_macros()
        });

        let res = match res {
            Ok(res) => res,
            Err(UnresolvedMacro { path }) => {
                if record_diagnostics {
                    self.source_map.diagnostics.push(BodyDiagnostic::UnresolvedMacroCall {
                        node: InFile::new(outer_file, syntax_ptr),
                        path,
                    });
                }
                return collector(self, None);
            }
        };

        if record_diagnostics {
            match &res.err {
                Some(ExpandError::UnresolvedProcMacro(krate)) => {
                    self.source_map.diagnostics.push(BodyDiagnostic::UnresolvedProcMacro {
                        node: InFile::new(outer_file, syntax_ptr),
                        krate: *krate,
                    });
                }
                Some(ExpandError::RecursionOverflowPoisoned) => {
                    // Recursion limit has been reached in the macro expansion tree, but not in
                    // this very macro call. Don't add diagnostics to avoid duplication.
                }
                Some(err) => {
                    self.source_map.diagnostics.push(BodyDiagnostic::MacroError {
                        node: InFile::new(outer_file, syntax_ptr),
                        message: err.to_string(),
                    });
                }
                None => {}
            }
        }

        match res.value {
            Some((mark, expansion)) => {
                // Keep collecting even with expansion errors so we can provide completions and
                // other services in incomplete macro expressions.
                self.source_map.expansions.insert(macro_call_ptr, self.expander.current_file_id);
                let prev_ast_id_map = mem::replace(
                    &mut self.ast_id_map,
                    self.db.ast_id_map(self.expander.current_file_id),
                );

                if record_diagnostics {
                    // FIXME: Report parse errors here
                }

                let id = collector(self, Some(expansion.tree()));
                self.ast_id_map = prev_ast_id_map;
                self.expander.exit(self.db, mark);
                id
            }
            None => collector(self, None),
        }
    }

    fn collect_expr_opt(&mut self, expr: Option<ast::Expr>) -> ExprId {
        let (expr, origin) = self.lower_expr_opt(expr);
        self.alloc_expr(expr, origin)
    }

    fn lower_expr_opt(&mut self, expr: Option<ast::Expr>) -> (Expr, ExprOrigin) {
        match expr {
            Some(expr) => self.lower_expr(expr),
            None => (Expr::Missing, ExprOrigin::None),
        }
    }

    fn collect_macro_as_stmt(
        &mut self,
        statements: &mut Vec<Statement>,
        mac: ast::MacroExpr,
    ) -> Option<ExprId> {
        let mac_call = mac.macro_call()?;
        let syntax_ptr = AstPtr::new(&ast::Expr::from(mac));
        let macro_ptr = AstPtr::new(&mac_call);
        let expansion = self.collect_macro_call(
            mac_call,
            macro_ptr,
            false,
            |this, expansion: Option<ast::MacroStmts>| match expansion {
                Some(expansion) => {
                    expansion.statements().for_each(|stmt| this.collect_stmt(statements, stmt));
                    expansion.expr().and_then(|expr| match expr {
                        ast::Expr::MacroExpr(mac) => this.collect_macro_as_stmt(statements, mac),
                        expr => Some(this.collect_expr(expr)),
                    })
                }
                None => None,
            },
        );
        match expansion {
            Some(tail) => {
                // Make the macro-call point to its expanded expression so we can query
                // semantics on syntax pointers to the macro
                let src = self.expander.to_source(syntax_ptr);
                self.source_map.expr_map.insert(src, tail);
                Some(tail)
            }
            None => None,
        }
    }

    fn collect_stmt(&mut self, statements: &mut Vec<Statement>, s: ast::Stmt) {
        match s {
            ast::Stmt::LetStmt(stmt) => {
                if self.check_cfg(&stmt).is_none() {
                    return;
                }
                let pat = self.collect_pat_top(stmt.pat());
                let type_ref =
                    stmt.ty().map(|it| Interned::new(TypeRef::from_ast(&self.ctx(), it)));
                let initializer = stmt.initializer().map(|e| self.collect_expr(e));
                let else_branch = stmt
                    .let_else()
                    .and_then(|let_else| let_else.block_expr())
                    .map(|block| self.collect_block(block));
                statements.push(Statement::Let { pat, type_ref, initializer, else_branch });
            }
            ast::Stmt::ExprStmt(stmt) => {
                let expr = stmt.expr();
                match &expr {
                    Some(expr) if self.check_cfg(expr).is_none() => return,
                    _ => (),
                }
                let has_semi = stmt.semicolon_token().is_some();
                // Note that macro could be expanded to multiple statements
                if let Some(ast::Expr::MacroExpr(mac)) = expr {
                    if let Some(expr) = self.collect_macro_as_stmt(statements, mac) {
                        statements.push(Statement::Expr { expr, has_semi })
                    }
                } else {
                    let expr = self.collect_expr_opt(expr);
                    statements.push(Statement::Expr { expr, has_semi });
                }
            }
            ast::Stmt::Item(_item) => (),
        }
    }

    fn collect_block(&mut self, block: ast::BlockExpr) -> ExprId {
        let (expr, origin) = self.lower_block(block);
        self.alloc_expr(expr, origin)
    }

    fn lower_block(&mut self, block: ast::BlockExpr) -> (Expr, ExprOrigin) {
        self.lower_block_(block, |id, statements, tail| Expr::Block {
            id,
            statements,
            tail,
            label: None,
        })
    }

    fn lower_block_(
        &mut self,
        block: ast::BlockExpr,
        mk_block: impl FnOnce(Option<BlockId>, StmtRange, Option<ExprId>) -> Expr,
    ) -> (Expr, ExprOrigin) {
        let block_has_items = {
            let statement_has_item = block.statements().any(|stmt| match stmt {
                ast::Stmt::Item(_) => true,
                // Macro calls can be both items and expressions. The syntax library always treats
                // them as expressions here, so we undo that.
                ast::Stmt::ExprStmt(es) => matches!(es.expr(), Some(ast::Expr::MacroExpr(_))),
                _ => false,
            });
            statement_has_item || matches!(block.tail_expr(), Some(ast::Expr::MacroExpr(_)))
        };

        let block_id = if block_has_items {
            let file_local_id = self.ast_id_map.ast_id(&block);
            let ast_id = AstId::new(self.expander.current_file_id, file_local_id);
            Some(self.db.intern_block(BlockLoc { ast_id, module: self.expander.module }))
        } else {
            None
        };

        let (module, def_map) =
            match block_id.map(|block_id| (self.db.block_def_map(block_id), block_id)) {
                Some((def_map, block_id)) => {
                    self.body.block_scopes.push(block_id);
                    (def_map.module_id(DefMap::ROOT), def_map)
                }
                None => (self.expander.module, self.def_map.clone()),
            };
        let prev_def_map = mem::replace(&mut self.def_map, def_map);
        let prev_local_module = mem::replace(&mut self.expander.module, module);

        let mut statements = Vec::new();
        block.statements().for_each(|s| self.collect_stmt(&mut statements, s));
        let tail = block.tail_expr().and_then(|e| match e {
            ast::Expr::MacroExpr(mac) => self.collect_macro_as_stmt(&mut statements, mac),
            expr => self.maybe_collect_expr(expr),
        });
        let tail = tail.or_else(|| {
            let stmt = statements.pop()?;
            if let Statement::Expr { expr, has_semi: false } = stmt {
                return Some(expr);
            }
            statements.push(stmt);
            None
        });

        let statements = self.body.stmts.alloc_many(statements);
        let syntax_node_ptr = AstPtr::new(&block.into());
        let result = (
            mk_block(block_id, statements, tail),
            ExprOrigin::User(self.expander.to_source(syntax_node_ptr)),
        );

        self.def_map = prev_def_map;
        self.expander.module = prev_local_module;
        result
    }

    fn collect_block_opt(&mut self, expr: Option<ast::BlockExpr>) -> ExprId {
        let (expr, origin) = self.lower_block_opt(expr);
        self.alloc_expr(expr, origin)
    }

    fn lower_block_opt(&mut self, expr: Option<ast::BlockExpr>) -> (Expr, ExprOrigin) {
        match expr {
            Some(block) => self.lower_block(block),
            None => (Expr::Missing, ExprOrigin::None),
        }
    }

    fn collect_labelled_block_opt(
        &mut self,
        label: Option<LabelId>,
        expr: Option<ast::BlockExpr>,
    ) -> ExprId {
        let (expr, origin) = self.lower_labelled_block_opt(label, expr);
        self.alloc_expr(expr, origin)
    }

    fn lower_labelled_block_opt(
        &mut self,
        label: Option<LabelId>,
        expr: Option<ast::BlockExpr>,
    ) -> (Expr, ExprOrigin) {
        match label {
            Some(label) => self.with_labeled_rib(label, |this| this.lower_block_opt(expr)),
            None => self.lower_block_opt(expr),
        }
    }

    // region: patterns

    fn collect_pat_top(&mut self, pat: Option<ast::Pat>) -> PatId {
        let (pat, origin, binding_id) = self.lower_pat_top(pat);
        self.alloc_pat(pat, origin, binding_id)
    }

    fn lower_pat_top(&mut self, pat: Option<ast::Pat>) -> (Pat, PatOrigin, Option<BindingId>) {
        match pat {
            Some(pat) => self.lower_pat(pat, &mut BindingList::default()),
            None => (Pat::Missing, PatOrigin::None, None),
        }
    }

    fn lower_pat(
        &mut self,
        pat: ast::Pat,
        binding_list: &mut BindingList,
    ) -> (Pat, PatOrigin, Option<BindingId>) {
        let pattern = match &pat {
            ast::Pat::IdentPat(bp) => return self.lower_ident_pat(&pat, bp, binding_list),
            ast::Pat::TupleStructPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let (args, ellipsis) = self.collect_tuple_pat(
                    p.fields(),
                    comma_follows_token(p.l_paren_token()),
                    binding_list,
                );
                Pat::TupleStruct { path, args, ellipsis }
            }
            ast::Pat::RefPat(p) => {
                let pat = self.collect_pat_opt(p.pat(), binding_list);
                let mutability = Mutability::from_mutable(p.mut_token().is_some());
                Pat::Ref { pat, mutability }
            }
            ast::Pat::PathPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                path.map(Pat::Path).unwrap_or(Pat::Missing)
            }
            ast::Pat::OrPat(p) => 'b: {
                let prev_is_used = mem::take(&mut binding_list.is_used);
                let prev_reject_new = mem::take(&mut binding_list.reject_new);
                let mut pats = Vec::with_capacity(p.pats().count());
                let mut it = p.pats();
                let Some(first) = it.next() else {
                    break 'b Pat::Or(PatRange::empty());
                };
                pats.push(self.lower_pat(first, binding_list));
                binding_list.reject_new = true;
                for rest in it {
                    for (_, x) in binding_list.is_used.iter_mut() {
                        *x = false;
                    }
                    pats.push(self.lower_pat(rest, binding_list));
                    for (&id, &x) in binding_list.is_used.iter() {
                        if !x {
                            self.body.bindings[id].problems =
                                Some(BindingProblems::NotBoundAcrossAll);
                        }
                    }
                }
                binding_list.reject_new = prev_reject_new;
                let current_is_used = mem::replace(&mut binding_list.is_used, prev_is_used);
                for (id, _) in current_is_used.into_iter() {
                    binding_list.check_is_used(self, id);
                }
                Pat::Or(self.alloc_pats(pats))
            }
            ast::Pat::ParenPat(p) => return self.lower_pat_opt(p.pat(), binding_list),
            ast::Pat::TuplePat(p) => {
                let (args, ellipsis) = self.collect_tuple_pat(
                    p.fields(),
                    comma_follows_token(p.l_paren_token()),
                    binding_list,
                );
                Pat::Tuple { args, ellipsis }
            }
            ast::Pat::WildcardPat(_) => Pat::Wild,
            ast::Pat::RecordPat(p) => {
                let path =
                    p.path().and_then(|path| self.expander.parse_path(self.db, path)).map(Box::new);
                let args: Vec<_> = p
                    .record_pat_field_list()
                    .expect("every struct should have a field list")
                    .fields()
                    .filter_map(|f| {
                        let ast_pat = f.pat()?;
                        let pat = self.collect_pat(ast_pat, binding_list);
                        let name = f.field_name()?.as_name();
                        Some(RecordFieldPat { name, pat })
                    })
                    .collect();
                let args = self.body.record_field_pats.alloc_many(args);

                let ellipsis = p
                    .record_pat_field_list()
                    .expect("every struct should have a field list")
                    .rest_pat()
                    .is_some();

                Pat::Record { path, args, ellipsis }
            }
            ast::Pat::SlicePat(p) => {
                let SlicePatComponents { prefix, slice, suffix } = p.components();
                let mut pats = Vec::with_capacity(std::cmp::max(prefix.len(), suffix.len()));

                pats.extend(prefix.into_iter().map(|p| self.lower_pat(p, binding_list)));
                let prefix = self.alloc_pats(pats.drain(..));

                let slice = slice.map(|p| self.collect_pat(p, binding_list));

                pats.extend(suffix.into_iter().map(|p| self.lower_pat(p, binding_list)));
                let suffix = self.alloc_pats(pats.drain(..));

                // FIXME properly handle `RestPat`
                Pat::Slice { prefix, slice, suffix }
            }
            #[rustfmt::skip] // https://github.com/rust-lang/rustfmt/issues/5676
            ast::Pat::LiteralPat(lit) => 'b: {
                let Some((hir_lit, ast_lit)) = pat_literal_to_hir(lit) else { break 'b Pat::Missing };
                let expr = Expr::Literal(hir_lit);
                let expr_ptr = AstPtr::new(&ast::Expr::Literal(ast_lit));
                let expr_id = self.alloc_expr(expr, ExprOrigin::User(self.expander.to_source(expr_ptr)));
                Pat::Lit(expr_id)
            }
            ast::Pat::RestPat(_) => {
                // `RestPat` requires special handling and should not be mapped
                // to a Pat. Here we are using `Pat::Missing` as a fallback for
                // when `RestPat` is mapped to `Pat`, which can easily happen
                // when the source code being analyzed has a malformed pattern
                // which includes `..` in a place where it isn't valid.

                Pat::Missing
            }
            ast::Pat::BoxPat(boxpat) => {
                let inner = self.collect_pat_opt(boxpat.pat(), binding_list);
                Pat::Box { inner }
            }
            ast::Pat::ConstBlockPat(const_block_pat) => {
                if let Some(block) = const_block_pat.block_expr() {
                    let expr_id = self.with_label_rib(RibKind::Constant, |this| {
                        let syntax_ptr = AstPtr::new(&block.clone().into());
                        let prev_binding_owner = this.current_binding_owner.take();
                        this.current_binding_owner = Some(this.expander.to_source(syntax_ptr));
                        let result = this.collect_block(block);
                        this.current_binding_owner = prev_binding_owner;
                        result
                    });
                    Pat::ConstBlock(expr_id)
                } else {
                    Pat::Missing
                }
            }
            ast::Pat::MacroPat(mac) => match mac.macro_call() {
                Some(call) => {
                    let macro_ptr = AstPtr::new(&call);
                    let src = self.expander.to_source(Either::Left(AstPtr::new(&pat)));
                    let (pat, parent, binding_id) =
                        self.collect_macro_call(call, macro_ptr, true, |this, expanded_pat| {
                            this.lower_pat_opt(expanded_pat, binding_list)
                        });
                    return (pat, PatOrigin::Macro(src, Box::new(parent)), binding_id);
                }
                None => Pat::Missing,
            },
            // FIXME: implement in a way that also builds source map and calculates assoc resolutions in type inference.
            ast::Pat::RangePat(p) => {
                let mut range_part_lower = |p: Option<ast::Pat>| {
                    p.and_then(|x| match &x {
                        ast::Pat::LiteralPat(x) => {
                            Some(Box::new(LiteralOrConst::Literal(pat_literal_to_hir(x)?.0)))
                        }
                        ast::Pat::IdentPat(p) => {
                            let name =
                                p.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);
                            Some(Box::new(LiteralOrConst::Const(name.into())))
                        }
                        ast::Pat::PathPat(p) => p
                            .path()
                            .and_then(|path| self.expander.parse_path(self.db, path))
                            .map(LiteralOrConst::Const)
                            .map(Box::new),
                        _ => None,
                    })
                };
                let start = range_part_lower(p.start());
                let end = range_part_lower(p.end());
                Pat::Range { start, end }
            }
        };
        let ptr = AstPtr::new(&pat);
        (pattern, PatOrigin::User(self.expander.to_source(Either::Left(ptr))), None)
    }

    fn collect_pat(&mut self, pat: ast::Pat, binding_list: &mut BindingList) -> PatId {
        let (pattern, ptr, binding_id) = self.lower_pat(pat, binding_list);
        self.alloc_pat(pattern, ptr, binding_id)
    }

    fn collect_pat_opt(&mut self, pat: Option<ast::Pat>, binding_list: &mut BindingList) -> PatId {
        let (pat, origin, binding_id) = self.lower_pat_opt(pat, binding_list);
        self.alloc_pat(pat, origin, binding_id)
    }

    fn lower_pat_opt(
        &mut self,
        pat: Option<ast::Pat>,
        binding_list: &mut BindingList,
    ) -> (Pat, PatOrigin, Option<BindingId>) {
        match pat {
            Some(pat) => self.lower_pat(pat, binding_list),
            None => Self::missing_pat(),
        }
    }

    fn lower_ident_pat(
        &mut self,
        pat: &ast::Pat,
        bp: &ast::IdentPat,
        binding_list: &mut BindingList,
    ) -> (Pat, PatOrigin, Option<BindingId>) {
        let name = bp.name().map(|nr| nr.as_name()).unwrap_or_else(Name::missing);

        let annotation = BindingAnnotation::new(bp.mut_token().is_some(), bp.ref_token().is_some());
        let subpat = bp.pat().map(|subpat| self.collect_pat(subpat, binding_list));

        let is_simple_ident_pat = annotation == BindingAnnotation::Unannotated && subpat.is_none();
        let (binding, pattern) = if is_simple_ident_pat {
            // This could also be a single-segment path pattern. To
            // decide that, we need to try resolving the name.
            let (resolved, _) = self.def_map.resolve_path(
                self.db,
                self.expander.module.local_id,
                &name.clone().into(),
                BuiltinShadowMode::Other,
                None,
            );
            match resolved.take_values() {
                Some(ModuleDefId::ConstId(_)) => (None, Pat::Path(name.into())),
                Some(ModuleDefId::EnumVariantId(_)) => {
                    // this is only really valid for unit variants, but
                    // shadowing other enum variants with a pattern is
                    // an error anyway
                    (None, Pat::Path(name.into()))
                }
                Some(ModuleDefId::AdtId(AdtId::StructId(s)))
                    if self.db.struct_data(s).variant_data.kind() != StructKind::Record =>
                {
                    // Funnily enough, record structs *can* be shadowed
                    // by pattern bindings (but unit or tuple structs
                    // can't).
                    (None, Pat::Path(name.into()))
                }
                // shadowing statics is an error as well, so we just ignore that case here
                _ => {
                    let id = binding_list.find(self, name, annotation);
                    (Some(id), Pat::Bind { id, subpat })
                }
            }
        } else {
            let id = binding_list.find(self, name, annotation);
            (Some(id), Pat::Bind { id, subpat })
        };

        let ptr = AstPtr::new(pat);
        (pattern, PatOrigin::User(self.expander.to_source(Either::Left(ptr))), binding)
    }

    fn collect_tuple_pat(
        &mut self,
        args: AstChildren<ast::Pat>,
        has_leading_comma: bool,
        binding_list: &mut BindingList,
    ) -> (PatRange, Option<u32>) {
        // Find the location of the `..`, if there is one. Note that we do not
        // consider the possibility of there being multiple `..` here.
        let ellipsis =
            args.clone().position(|p| matches!(p, ast::Pat::RestPat(_))).map(|p| p as u32);
        // We want to skip the `..` pattern here, since we account for it above.
        let args = args
            .filter(|p| !matches!(p, ast::Pat::RestPat(_)))
            .map(|p| self.lower_pat(p, binding_list));
        // if there is a leading comma, the user is most likely to type out a leading pattern
        // so we insert a missing pattern at the beginning for IDE features
        let first =
            if has_leading_comma { Some((Pat::Missing, PatOrigin::None, None)) } else { None };

        let args: Vec<_> = first.into_iter().chain(args).collect();
        (self.alloc_pats(args), ellipsis)
    }

    // endregion: patterns

    /// Returns `None` (and emits diagnostics) when `owner` if `#[cfg]`d out, and `Some(())` when
    /// not.
    fn check_cfg(&mut self, owner: &dyn ast::HasAttrs) -> Option<()> {
        match self.expander.parse_attrs(self.db, owner).cfg() {
            Some(cfg) => {
                if self.expander.cfg_options().check(&cfg) != Some(false) {
                    return Some(());
                }

                self.source_map.diagnostics.push(BodyDiagnostic::InactiveCode {
                    node: InFile::new(
                        self.expander.current_file_id,
                        SyntaxNodePtr::new(owner.syntax()),
                    ),
                    cfg,
                    opts: self.expander.cfg_options().clone(),
                });

                None
            }
            None => Some(()),
        }
    }

    // region: labels

    fn collect_label(&mut self, ast_label: ast::Label) -> LabelId {
        let label = Label {
            name: ast_label.lifetime().as_ref().map_or_else(Name::missing, Name::new_lifetime),
        };
        self.alloc_label(label, AstPtr::new(&ast_label))
    }

    fn resolve_label(
        &self,
        lifetime: Option<ast::Lifetime>,
    ) -> Result<Option<LabelId>, BodyDiagnostic> {
        let Some(lifetime) = lifetime else {
            return Ok(None)
        };
        let name = Name::new_lifetime(&lifetime);

        for (rib_idx, rib) in self.label_ribs.iter().enumerate().rev() {
            if let Some((label_name, id)) = &rib.label {
                if *label_name == name {
                    return if self.is_label_valid_from_rib(rib_idx) {
                        Ok(Some(*id))
                    } else {
                        Err(BodyDiagnostic::UnreachableLabel {
                            name,
                            node: InFile::new(
                                self.expander.current_file_id,
                                AstPtr::new(&lifetime),
                            ),
                        })
                    };
                }
            }
        }

        Err(BodyDiagnostic::UndeclaredLabel {
            name,
            node: InFile::new(self.expander.current_file_id, AstPtr::new(&lifetime)),
        })
    }

    fn is_label_valid_from_rib(&self, rib_index: usize) -> bool {
        !self.label_ribs[rib_index + 1..].iter().any(|rib| rib.kind.is_label_barrier())
    }

    fn with_label_rib<T>(&mut self, kind: RibKind, f: impl FnOnce(&mut Self) -> T) -> T {
        self.label_ribs.push(LabelRib::new(kind));
        let res = f(self);
        self.label_ribs.pop();
        res
    }

    fn with_labeled_rib<T>(&mut self, label: LabelId, f: impl FnOnce(&mut Self) -> T) -> T {
        self.label_ribs.push(LabelRib::new_normal((self.body[label].name.clone(), label)));
        let res = f(self);
        self.label_ribs.pop();
        res
    }

    fn with_opt_labeled_rib<T>(
        &mut self,
        label: Option<LabelId>,
        f: impl FnOnce(&mut Self) -> T,
    ) -> T {
        match label {
            None => f(self),
            Some(label) => self.with_labeled_rib(label, f),
        }
    }
    // endregion: labels
}

fn pat_literal_to_hir(lit: &ast::LiteralPat) -> Option<(Literal, ast::Literal)> {
    let ast_lit = lit.literal()?;
    let mut hir_lit: Literal = ast_lit.kind().into();
    if lit.minus_token().is_some() {
        let Some(h) = hir_lit.negate() else {
            return None;
        };
        hir_lit = h;
    }
    Some((hir_lit, ast_lit))
}

impl ExprCollector<'_> {
    fn alloc_expr(&mut self, expr: Expr, mut origin: ExprOrigin) -> ExprId {
        let expr_id = self.body.exprs.alloc(expr);
        loop {
            match origin {
                ExprOrigin::None => break,
                ExprOrigin::User(src) => {
                    self.source_map.expr_map_back.insert(expr_id, src.clone());
                    self.source_map.expr_map.insert(src, expr_id);
                    break;
                }
                ExprOrigin::Macro(macro_src, parent_origin) => {
                    self.source_map.expr_map.insert(macro_src, expr_id);
                    origin = *parent_origin;
                }
            }
        }
        expr_id
    }

    fn alloc_exprs(&mut self, exprs: impl IntoIterator<Item = (Expr, ExprOrigin)>) -> ExprRange {
        let start = self.body.exprs.next_idx();
        for (expr, origin) in exprs {
            self.alloc_expr(expr, origin);
        }
        let end = self.body.exprs.next_idx();
        ExprRange::new(start..end)
    }

    // FIXME: desugared exprs don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_expr_desugared(&mut self, expr: Expr) -> ExprId {
        self.body.exprs.alloc(expr)
    }

    fn missing_expr(&mut self) -> ExprId {
        self.alloc_expr_desugared(Expr::Missing)
    }

    fn alloc_binding(&mut self, name: Name, mode: BindingAnnotation) -> BindingId {
        let binding_id = self.body.bindings.alloc(Binding {
            name,
            mode,
            definitions: SmallVec::new(),
            owner: None,
            problems: None,
        });
        if let Some(owner) = &self.current_binding_owner {
            self.binding_owners.insert(binding_id, owner.clone());
        }
        binding_id
    }

    fn alloc_pat(
        &mut self,
        pat: Pat,
        mut origin: PatOrigin,
        binding_id: Option<BindingId>,
    ) -> PatId {
        let pat_id = self.body.pats.alloc(pat);
        loop {
            match origin {
                PatOrigin::None => break,
                PatOrigin::User(src) => {
                    self.source_map.pat_map_back.insert(pat_id, src.clone());
                    self.source_map.pat_map.insert(src, pat_id);
                    break;
                }
                PatOrigin::Macro(macro_src, parent_origin) => {
                    self.source_map.pat_map.insert(macro_src, pat_id);
                    origin = *parent_origin;
                }
            }
        }

        if let Some(binding_id) = binding_id {
            self.body.bindings[binding_id].definitions.push(pat_id);
        }
        pat_id
    }

    // FIXME: desugared pats don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_pat_desugared(&mut self, pat: Pat) -> PatId {
        self.alloc_pat(pat, PatOrigin::None, None)
    }

    fn missing_pat() -> (Pat, PatOrigin, Option<BindingId>) {
        (Pat::Missing, PatOrigin::None, None)
    }

    fn alloc_pats(
        &mut self,
        pats: impl IntoIterator<Item = (Pat, PatOrigin, Option<BindingId>)>,
    ) -> PatRange {
        let start = self.body.pats.next_idx();
        for (pat, origin, binding_id) in pats {
            self.alloc_pat(pat, origin, binding_id);
        }
        let end = self.body.pats.next_idx();
        PatRange::new(start..end)
    }

    fn alloc_label(&mut self, label: Label, ptr: LabelPtr) -> LabelId {
        let src = self.expander.to_source(ptr);
        let id = self.body.labels.alloc(label);
        self.source_map.label_map_back.insert(id, src.clone());
        self.source_map.label_map.insert(src, id);
        id
    }
    // FIXME: desugared labels don't have ptr, that's wrong and should be fixed somehow.
    fn alloc_label_desugared(&mut self, label: Label) -> LabelId {
        self.body.labels.alloc(label)
    }
}

#[derive(Debug, Clone)]
enum LowerOrigin<T> {
    None,
    User(InFile<T>),
    Macro(InFile<T>, Box<Self>), // FIXME: use Vec/SmallVec instead of linked list
}

type ExprOrigin = LowerOrigin<ExprPtr>;
type PatOrigin = LowerOrigin<PatPtr>;

fn comma_follows_token(t: Option<syntax::SyntaxToken>) -> bool {
    (|| syntax::algo::skip_trivia_token(t?.next_token()?, syntax::Direction::Next))()
        .map_or(false, |it| it.kind() == syntax::T![,])
}
