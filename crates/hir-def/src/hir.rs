//! This module describes hir-level representation of expressions.
//!
//! This representation is:
//!
//! 1. Identity-based. Each expression has an `id`, so we can distinguish
//!    between different `1` in `1 + 1`.
//! 2. Independent of syntax. Though syntactic provenance information can be
//!    attached separately via id-based side map.
//! 3. Unresolved. Paths are stored as sequences of names, and not as defs the
//!    names refer to.
//! 4. Desugared. There's no `if let`.
//!
//! See also a neighboring `body` module.

pub mod type_ref;

use std::{fmt, num::NonZeroU32};

use hir_expand::name::Name;
use intern::Interned;
use la_arena::{Idx, IdxRange, RawIdx};
use smallvec::SmallVec;
use syntax::ast::{self};

use crate::{
    body::Body,
    builtin_type::{BuiltinFloat, BuiltinInt, BuiltinUint},
    path::{GenericArgs, Path},
    type_ref::{Mutability, Rawness, TypeRef},
    AnonymousConstId, BlockId,
};

pub use syntax::ast::{ArithOp, BinaryOp, CmpOp, LogicOp, Ordering, RangeOp, UnaryOp};

pub type BindingId = Idx<Binding>;

pub type ExprId = Idx<Expr>;
pub type ExprRange = IdxRange<Expr>;

/// FIXME: this is a hacky function which should be removed
pub(crate) fn dummy_expr_id() -> ExprId {
    ExprId::from_raw(RawIdx::from(u32::MAX))
}

pub type PatId = Idx<Pat>;
pub type PatRange = IdxRange<Pat>;

pub type StmtId = Idx<Statement>;
pub type StmtRange = IdxRange<Statement>;

#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum ExprOrPatId {
    ExprId(ExprId),
    PatId(PatId),
}
stdx::impl_from!(ExprId, PatId for ExprOrPatId);

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Label {
    pub name: Name,
}
pub type LabelId = Idx<Label>;

// We convert float values into bits and that's how we don't need to deal with f32 and f64.
// For PartialEq, bits comparison should work, as ordering is not important
// https://github.com/rust-lang/rust-analyzer/issues/12380#issuecomment-1137284360
#[derive(Default, Debug, Clone, Copy, Eq, PartialEq)]
pub struct FloatTypeWrapper(u64);

impl FloatTypeWrapper {
    pub fn new(value: f64) -> Self {
        Self(value.to_bits())
    }

    pub fn into_f64(self) -> f64 {
        f64::from_bits(self.0)
    }

    pub fn into_f32(self) -> f32 {
        f64::from_bits(self.0) as f32
    }
}

impl fmt::Display for FloatTypeWrapper {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", f64::from_bits(self.0))
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Literal {
    String(Box<str>),
    ByteString(Box<[u8]>),
    CString(Box<str>),
    Char(char),
    Bool(bool),
    Int(i128, Option<BuiltinInt>),
    Uint(u128, Option<BuiltinUint>),
    // Here we are using a wrapper around float because f32 and f64 do not implement Eq, so they
    // could not be used directly here, to understand how the wrapper works go to definition of
    // FloatTypeWrapper
    Float(FloatTypeWrapper, Option<BuiltinFloat>),
}

#[derive(Debug, Clone, Eq, PartialEq)]
/// Used in range patterns.
pub enum LiteralOrConst {
    Literal(Literal),
    Const(Path),
}

impl Literal {
    pub fn negate(self) -> Option<Self> {
        if let Literal::Int(i, k) = self {
            Some(Literal::Int(-i, k))
        } else {
            None
        }
    }
}

impl From<ast::LiteralKind> for Literal {
    fn from(ast_lit_kind: ast::LiteralKind) -> Self {
        use ast::LiteralKind;
        match ast_lit_kind {
            // FIXME: these should have actual values filled in, but unsure on perf impact
            LiteralKind::IntNumber(lit) => {
                if let builtin @ Some(_) = lit.suffix().and_then(BuiltinFloat::from_suffix) {
                    Literal::Float(
                        FloatTypeWrapper::new(lit.float_value().unwrap_or(Default::default())),
                        builtin,
                    )
                } else if let builtin @ Some(_) = lit.suffix().and_then(BuiltinUint::from_suffix) {
                    Literal::Uint(lit.value().unwrap_or(0), builtin)
                } else {
                    let builtin = lit.suffix().and_then(BuiltinInt::from_suffix);
                    Literal::Int(lit.value().unwrap_or(0) as i128, builtin)
                }
            }
            LiteralKind::FloatNumber(lit) => {
                let ty = lit.suffix().and_then(BuiltinFloat::from_suffix);
                Literal::Float(FloatTypeWrapper::new(lit.value().unwrap_or(Default::default())), ty)
            }
            LiteralKind::ByteString(bs) => {
                let text = bs.value().map(Box::from).unwrap_or_else(Default::default);
                Literal::ByteString(text)
            }
            LiteralKind::String(s) => {
                let text = s.value().map(Box::from).unwrap_or_else(Default::default);
                Literal::String(text)
            }
            LiteralKind::CString(s) => {
                let text = s.value().map(Box::from).unwrap_or_else(Default::default);
                Literal::CString(text)
            }
            LiteralKind::Byte(b) => {
                Literal::Uint(b.value().unwrap_or_default() as u128, Some(BuiltinUint::U8))
            }
            LiteralKind::Char(c) => Literal::Char(c.value().unwrap_or_default()),
            LiteralKind::Bool(val) => Literal::Bool(val),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Expr {
    /// This is produced if the syntax tree does not have a required expression piece.
    Missing,
    Path(Path),
    If {
        condition: ExprId,
        then_branch: ExprId,
        else_branch: Option<ExprId>,
    },
    Let {
        pat: PatId,
        expr: ExprId,
    },
    Block {
        id: Option<BlockId>,
        statements: StmtRange,
        tail: Option<ExprId>,
        label: Option<LabelId>,
    },
    Async {
        id: Option<BlockId>,
        statements: StmtRange,
        tail: Option<ExprId>,
    },
    Const(AnonymousConstId),
    Unsafe {
        id: Option<BlockId>,
        statements: StmtRange,
        tail: Option<ExprId>,
    },
    Loop {
        body: ExprId,
        label: Option<LabelId>,
    },
    While {
        condition: ExprId,
        body: ExprId,
        label: Option<LabelId>,
    },
    Call {
        callee: ExprId,
        args: ExprRange,
        is_assignee_expr: bool,
    },
    MethodCall {
        receiver: ExprId,
        method_name: Name,
        args: ExprRange,
        generic_args: Option<Box<GenericArgs>>,
    },
    Match {
        expr: ExprId,
        arms: Box<[MatchArm]>,
    },
    Continue {
        label: Option<LabelId>,
    },
    Break {
        expr: Option<ExprId>,
        label: Option<LabelId>,
    },
    Return {
        expr: Option<ExprId>,
    },
    Yield {
        expr: Option<ExprId>,
    },
    Yeet {
        expr: Option<ExprId>,
    },
    RecordLit {
        path: Option<Box<Path>>,
        fields: Box<[RecordLitField]>,
        spread: Option<ExprId>,
        ellipsis: bool,
        is_assignee_expr: bool,
    },
    Field {
        expr: ExprId,
        name: Name,
    },
    Await {
        expr: ExprId,
    },
    Cast {
        expr: ExprId,
        type_ref: Interned<TypeRef>,
    },
    Ref {
        expr: ExprId,
        rawness: Rawness,
        mutability: Mutability,
    },
    Box {
        expr: ExprId,
    },
    UnaryOp {
        expr: ExprId,
        op: UnaryOp,
    },
    BinaryOp {
        lhs: ExprId,
        rhs: ExprId,
        op: Option<BinaryOp>,
    },
    Range {
        lhs: Option<ExprId>,
        rhs: Option<ExprId>,
        range_type: RangeOp,
    },
    Index {
        base: ExprId,
        index: ExprId,
    },
    Closure {
        args: PatRange,
        arg_types: Box<[Option<Interned<TypeRef>>]>,
        ret_type: Option<Interned<TypeRef>>,
        body: ExprId,
        closure_kind: ClosureKind,
        capture_by: CaptureBy,
    },
    Tuple {
        exprs: ExprRange,
        is_assignee_expr: bool,
    },
    Array(Array),
    Literal(Literal),
    Underscore,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClosureKind {
    Closure,
    Generator(Movability),
    Async,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureBy {
    /// `move |x| y + x`.
    Value,
    /// `move` keyword was not specified.
    Ref,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Movability {
    Static,
    Movable,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Array {
    ElementList { elements: ExprRange, is_assignee_expr: bool },
    Repeat { initializer: ExprId, repeat: ExprId },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MatchArm {
    pub pat: PatId,
    pub guard: Option<ExprId>,
    pub expr: ExprId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RecordLitField {
    pub name: Name,
    pub expr: ExprId,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Statement {
    Let {
        pat: PatId,
        type_ref: Option<Interned<TypeRef>>,
        initializer: Option<ExprId>,
        else_branch: Option<ExprId>,
    },
    Expr {
        expr: ExprId,
        has_semi: bool,
    },
}

impl Expr {
    pub fn walk_child_exprs(&self, body: &Body, mut f: impl FnMut(ExprId)) {
        match self {
            Expr::Missing => {}
            Expr::Path(_) => {}
            Expr::If { condition, then_branch, else_branch } => {
                f(*condition);
                f(*then_branch);
                if let &Some(else_branch) = else_branch {
                    f(else_branch);
                }
            }
            Expr::Let { expr, .. } => {
                f(*expr);
            }
            Expr::Const(_) => (),
            Expr::Block { statements, tail, .. }
            | Expr::Unsafe { statements, tail, .. }
            | Expr::Async { statements, tail, .. } => {
                for stmt in &body[*statements] {
                    match stmt {
                        Statement::Let { initializer, else_branch, .. } => {
                            if let Some(expr) = initializer {
                                f(*expr);
                            }
                            if let Some(expr) = else_branch {
                                f(*expr);
                            }
                        }
                        Statement::Expr { expr: expression, .. } => f(*expression),
                    }
                }
                if let &Some(expr) = tail {
                    f(expr);
                }
            }
            Expr::Loop { body, .. } => f(*body),
            Expr::While { condition, body, .. } => {
                f(*condition);
                f(*body);
            }
            Expr::Call { callee, args, .. } => {
                f(*callee);
                args.iter().for_each(f);
            }
            Expr::MethodCall { receiver, args, .. } => {
                f(*receiver);
                args.iter().for_each(f);
            }
            Expr::Match { expr, arms } => {
                f(*expr);
                arms.iter().map(|arm| arm.expr).for_each(f);
            }
            Expr::Continue { .. } => {}
            Expr::Break { expr, .. }
            | Expr::Return { expr }
            | Expr::Yield { expr }
            | Expr::Yeet { expr } => {
                if let &Some(expr) = expr {
                    f(expr);
                }
            }
            Expr::RecordLit { fields, spread, .. } => {
                for field in fields.iter() {
                    f(field.expr);
                }
                if let &Some(expr) = spread {
                    f(expr);
                }
            }
            Expr::Closure { body, .. } => {
                f(*body);
            }
            Expr::BinaryOp { lhs, rhs, .. } => {
                f(*lhs);
                f(*rhs);
            }
            Expr::Range { lhs, rhs, .. } => {
                if let &Some(lhs) = rhs {
                    f(lhs);
                }
                if let &Some(rhs) = lhs {
                    f(rhs);
                }
            }
            Expr::Index { base, index } => {
                f(*base);
                f(*index);
            }
            Expr::Field { expr, .. }
            | Expr::Await { expr }
            | Expr::Cast { expr, .. }
            | Expr::Ref { expr, .. }
            | Expr::UnaryOp { expr, .. }
            | Expr::Box { expr } => {
                f(*expr);
            }
            Expr::Tuple { exprs, .. } => exprs.iter().for_each(f),
            Expr::Array(a) => match a {
                Array::ElementList { elements, .. } => elements.iter().for_each(f),
                Array::Repeat { initializer, repeat } => {
                    f(*initializer);
                    f(*repeat)
                }
            },
            Expr::Literal(_) => {}
            Expr::Underscore => {}
        }
    }
}

/// Explicit binding annotations given in the HIR for a binding. Note
/// that this is not the final binding *mode* that we infer after type
/// inference.
#[derive(Clone, PartialEq, Eq, Debug, Copy)]
pub enum BindingAnnotation {
    /// No binding annotation given: this means that the final binding mode
    /// will depend on whether we have skipped through a `&` reference
    /// when matching. For example, the `x` in `Some(x)` will have binding
    /// mode `None`; if you do `let Some(x) = &Some(22)`, it will
    /// ultimately be inferred to be by-reference.
    Unannotated,

    /// Annotated with `mut x` -- could be either ref or not, similar to `None`.
    Mutable,

    /// Annotated as `ref`, like `ref x`
    Ref,

    /// Annotated as `ref mut x`.
    RefMut,
}

impl BindingAnnotation {
    pub fn new(is_mutable: bool, is_ref: bool) -> Self {
        match (is_mutable, is_ref) {
            (true, true) => BindingAnnotation::RefMut,
            (false, true) => BindingAnnotation::Ref,
            (true, false) => BindingAnnotation::Mutable,
            (false, false) => BindingAnnotation::Unannotated,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum BindingProblems {
    /// https://doc.rust-lang.org/stable/error_codes/E0416.html
    BoundMoreThanOnce,
    /// https://doc.rust-lang.org/stable/error_codes/E0409.html
    BoundInconsistently,
    /// https://doc.rust-lang.org/stable/error_codes/E0408.html
    NotBoundAcrossAll,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Binding {
    pub name: Name,
    pub mode: BindingAnnotation,
    pub definitions: SmallVec<[PatId; 1]>,
    /// Id of the closure/generator that owns this binding. If it is owned by the
    /// top level expression, this field would be `None`.
    pub owner: Option<ExprId>,
    pub problems: Option<BindingProblems>,
}

impl Binding {
    pub fn is_upvar(&self, relative_to: ExprId) -> bool {
        match self.owner {
            Some(x) => {
                // We assign expression ids in a way that outer closures will receive
                // a higher id
                x.into_raw() > relative_to.into_raw()
            }
            None => true,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RecordFieldPat {
    pub name: Name,
    pub pat: PatId,
}

pub type RecordFieldPatIdx = Idx<RecordFieldPat>;
pub type RecordFieldPatRange = IdxRange<RecordFieldPat>;

/// Close relative to rustc's hir::PatKind
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum Pat {
    Missing,
    Wild,
    Tuple { args: PatRange, ellipsis: Option<Ellipsis> },
    Or(PatRange),
    Record { path: Option<Box<Path>>, args: RecordFieldPatRange, ellipsis: bool },
    Range { start: Option<Box<LiteralOrConst>>, end: Option<Box<LiteralOrConst>> },
    Slice { prefix: PatRange, slice: Option<PatId>, suffix: PatRange },
    Path(Box<Path>),
    Lit(ExprId),
    Bind { id: BindingId, subpat: Option<PatId> },
    TupleStruct { path: Option<Box<Path>>, args: PatRange, ellipsis: Option<Ellipsis> },
    Ref { pat: PatId, mutability: Mutability },
    Box { inner: PatId },
    ConstBlock(ExprId),
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Ellipsis(NonZeroU32);

impl From<usize> for Ellipsis {
    fn from(value: usize) -> Self {
        unsafe { Self(NonZeroU32::new_unchecked((value as u32).saturating_add(1))) }
    }
}

impl From<Ellipsis> for usize {
    fn from(value: Ellipsis) -> Self {
        value.0.get().saturating_sub(1) as usize
    }
}

impl Pat {
    pub fn walk_child_pats(&self, body: &Body, mut f: impl FnMut(PatId)) {
        match self {
            Pat::Range { .. }
            | Pat::Lit(..)
            | Pat::Path(..)
            | Pat::ConstBlock(..)
            | Pat::Wild
            | Pat::Missing => {}
            Pat::Bind { subpat, .. } => {
                subpat.iter().copied().for_each(f);
            }
            Pat::Or(args) | Pat::Tuple { args, .. } | Pat::TupleStruct { args, .. } => {
                args.iter().for_each(f);
            }
            Pat::Ref { pat, .. } => f(*pat),
            Pat::Slice { prefix, slice, suffix } => {
                prefix.iter().chain(slice.iter().copied()).chain(suffix.iter()).for_each(f)
            }
            Pat::Record { args, .. } => {
                body[*args].iter().for_each(|RecordFieldPat { pat, .. }| f(*pat))
            }
            Pat::Box { inner } => f(*inner),
        }
    }
}

#[cfg(test)]
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_tests {
    use super::*;

    #[test]
    fn expr_size() {
        assert_eq!(std::mem::size_of::<Expr>(), 48);
    }

    #[test]
    fn pat_size() {
        assert_eq!(std::mem::size_of::<Pat>(), 24);
    }

    #[test]
    fn literal_size() {
        assert_eq!(std::mem::size_of::<Literal>(), 24);
    }

    #[test]
    fn statement_size() {
        assert_eq!(std::mem::size_of::<Statement>(), 24);
    }
}
