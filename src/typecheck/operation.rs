//! Typing of primitive operations.
use super::*;
use crate::{
    error::TypecheckError,
    label::{Polarity, TypeVarData},
    term::{BinaryOp, NAryOp, RecordExtKind, UnaryOp},
    types::TypeF,
};
use crate::{mk_uty_arrow, mk_uty_enum, mk_uty_record};

/// Type of unary operations.
pub fn get_uop_type(
    state: &mut State,
    op: &UnaryOp,
) -> Result<(UnifType, UnifType), TypecheckError> {
    Ok(match op {
        // forall a. bool -> a -> a -> a
        UnaryOp::Ite() => {
            let branches = UnifType::UnifVar(state.table.fresh_type_var_id());

            (
                mk_uniftype::bool(),
                mk_uty_arrow!(branches.clone(), branches.clone(), branches),
            )
        }
        // Dyn -> [| 'Number, 'Bool, 'String, 'Enum, 'Function, 'Array, 'Record, 'Label, 'Other |]
        UnaryOp::Typeof() => (
            mk_uniftype::dynamic(),
            mk_uty_enum!(
                "Number", "Bool", "String", "Enum", "Function", "Array", "Record", "Label", "Other"
            ),
        ),
        // Bool -> Bool -> Bool
        UnaryOp::BoolAnd() | UnaryOp::BoolOr() => {
            (mk_uniftype::bool(), mk_uty_arrow!(TypeF::Bool, TypeF::Bool))
        }
        // Bool -> Bool
        UnaryOp::BoolNot() => (mk_uniftype::bool(), mk_uniftype::bool()),
        // forall a. Dyn -> a
        UnaryOp::Blame() => {
            let res = UnifType::UnifVar(state.table.fresh_type_var_id());

            (mk_uniftype::dynamic(), res)
        }
        // Dyn -> Polarity
        UnaryOp::Pol() => (mk_uniftype::dynamic(), mk_uty_enum!("Positive", "Negative")),
        // forall rows. < | rows> -> <id | rows>
        UnaryOp::Embed(id) => {
            let row_var_id = state.table.fresh_erows_var_id();
            let row = UnifEnumRows::UnifVar(row_var_id);

            let domain = mk_uty_enum!(; row.clone());
            let codomain = mk_uty_enum!(*id; row);

            // The codomain is the only type which can impose a constraint on the fresh row
            // unification variable, namely that it can't contain `id`.
            codomain.constrain_fresh_erows_var(state, row_var_id);
            (domain, codomain)
        }
        // This should not happen, as a match primop is only produced during evaluation.
        UnaryOp::Match { .. } => panic!("cannot typecheck match primop"),
        // Morally, Label -> Label
        // Dyn -> Dyn
        UnaryOp::ChangePolarity()
        | UnaryOp::GoDom()
        | UnaryOp::GoCodom()
        | UnaryOp::GoArray()
        | UnaryOp::GoDict() => (mk_uniftype::dynamic(), mk_uniftype::dynamic()),
        // forall rows a. { id: a | rows} -> a
        UnaryOp::StaticAccess(id) => {
            let rows = state.table.fresh_rrows_uvar();
            let res = state.table.fresh_type_uvar();

            (mk_uty_record!((*id, res.clone()); rows), res)
        }
        // forall a b. Array a -> (a -> b) -> Array b
        UnaryOp::ArrayMap() => {
            let a = UnifType::UnifVar(state.table.fresh_type_var_id());
            let b = UnifType::UnifVar(state.table.fresh_type_var_id());

            let f_type = mk_uty_arrow!(a.clone(), b.clone());
            (
                mk_uniftype::array(a),
                mk_uty_arrow!(f_type, mk_uniftype::array(b)),
            )
        }
        // forall a. Num -> (Num -> a) -> Array a
        UnaryOp::ArrayGen() => {
            let a = UnifType::UnifVar(state.table.fresh_type_var_id());

            let f_type = mk_uty_arrow!(TypeF::Number, a.clone());
            (
                mk_uniftype::num(),
                mk_uty_arrow!(f_type, mk_uniftype::array(a)),
            )
        }
        // forall a b. { _ : a} -> (Str -> a -> b) -> { _ : b }
        UnaryOp::RecordMap() => {
            // Assuming f has type Str -> a -> b,
            // this has type Dict(a) -> Dict(b)

            let a = UnifType::UnifVar(state.table.fresh_type_var_id());
            let b = UnifType::UnifVar(state.table.fresh_type_var_id());

            let f_type = mk_uty_arrow!(TypeF::String, a.clone(), b.clone());
            (
                mk_uniftype::dict(a),
                mk_uty_arrow!(f_type, mk_uniftype::dict(b)),
            )
        }
        // forall a b. a -> b -> b
        UnaryOp::Seq() | UnaryOp::DeepSeq() => {
            let fst = UnifType::UnifVar(state.table.fresh_type_var_id());
            let snd = UnifType::UnifVar(state.table.fresh_type_var_id());

            (fst, mk_uty_arrow!(snd.clone(), snd))
        }
        // forall a. Array a -> Num
        UnaryOp::ArrayLength() => {
            let ty_elt = UnifType::UnifVar(state.table.fresh_type_var_id());
            (mk_uniftype::array(ty_elt), mk_uniftype::num())
        }
        // This should not happen, as ChunksConcat() is only produced during evaluation.
        UnaryOp::ChunksConcat() => panic!("cannot type ChunksConcat()"),
        // forall a. { _: a } -> Array Str
        UnaryOp::FieldsOf() => {
            let ty_a = UnifType::UnifVar(state.table.fresh_type_var_id());

            (
                mk_uniftype::dict(ty_a),
                mk_uniftype::array(mk_uniftype::str()),
            )
        }
        // forall a. { _: a } -> Array a
        UnaryOp::ValuesOf() => {
            let ty_a = UnifType::UnifVar(state.table.fresh_type_var_id());

            (mk_uniftype::dict(ty_a.clone()), mk_uniftype::array(ty_a))
        }
        // Str -> Str
        UnaryOp::StrTrim() => (mk_uniftype::str(), mk_uniftype::str()),
        // Str -> Array Str
        UnaryOp::StrChars() => (mk_uniftype::str(), mk_uniftype::array(mk_uniftype::str())),
        // Str -> Str
        UnaryOp::StrUppercase() => (mk_uniftype::str(), mk_uniftype::str()),
        // Str -> Str
        UnaryOp::StrLowercase() => (mk_uniftype::str(), mk_uniftype::str()),
        // Str -> Num
        UnaryOp::StrLength() => (mk_uniftype::str(), mk_uniftype::num()),
        // Dyn -> Str
        UnaryOp::ToStr() => (mk_uniftype::dynamic(), mk_uniftype::str()),
        // Str -> Num
        UnaryOp::NumFromStr() => (mk_uniftype::str(), mk_uniftype::num()),
        // Str -> < | a> for a rigid type variable a
        UnaryOp::EnumFromStr() => (
            mk_uniftype::str(),
            mk_uty_enum!(; state.table.fresh_erows_const()),
        ),
        // Str -> Str -> Bool
        UnaryOp::StrIsMatch() => (
            mk_uniftype::str(),
            mk_uty_arrow!(mk_uniftype::str(), mk_uniftype::bool()),
        ),
        // Str -> Str -> {matched: Str, index: Num, groups: Array Str}
        UnaryOp::StrFind() => (
            mk_uniftype::str(),
            mk_uty_arrow!(
                mk_uniftype::str(),
                mk_uty_record!(
                    ("matched", TypeF::String),
                    ("index", TypeF::Number),
                    ("groups", mk_uniftype::array(TypeF::String))
                )
            ),
        ),
        // Str -> Bool
        UnaryOp::StrIsMatchCompiled(_) => (mk_uniftype::str(), mk_uniftype::bool()),
        // Str -> {matched: Str, index: Num, groups: Array Str}
        UnaryOp::StrFindCompiled(_) => (
            mk_uniftype::str(),
            mk_uty_record!(
                ("matched", TypeF::String),
                ("index", TypeF::Number),
                ("groups", mk_uniftype::array(TypeF::String))
            ),
        ),
        // Dyn -> Dyn
        UnaryOp::Force { .. } => (mk_uniftype::dynamic(), mk_uniftype::dynamic()),
        // forall a. a -> a
        UnaryOp::RecDefault() => {
            let ty = state.table.fresh_type_uvar();
            (ty.clone(), ty)
        }
        // forall a. a -> a
        UnaryOp::RecForce() => {
            let ty = state.table.fresh_type_uvar();
            (ty.clone(), ty)
        }
        UnaryOp::RecordEmptyWithTail() => (mk_uniftype::dynamic(), mk_uniftype::dynamic()),

        // forall a. Str -> a -> a
        UnaryOp::Trace() => {
            let ty = UnifType::UnifVar(state.table.fresh_type_var_id());
            (mk_uniftype::str(), mk_uty_arrow!(ty.clone(), ty))
        }
        // Morally: Lbl -> Lbl
        // Actual: Dyn -> Dyn
        UnaryOp::LabelPushDiag() => (mk_uniftype::dynamic(), mk_uniftype::dynamic()),
        // Morally: Lbl -> Bool
        // Actual: Dyn -> Bool
        UnaryOp::Dualize() => (mk_uniftype::dynamic(), mk_uniftype::bool()),
    })
}

/// Type of a binary operation.
pub fn get_bop_type(
    state: &mut State,
    op: &BinaryOp,
) -> Result<(UnifType, UnifType, UnifType), TypecheckError> {
    Ok(match op {
        // Num -> Num -> Num
        BinaryOp::Plus()
        | BinaryOp::Sub()
        | BinaryOp::Mult()
        | BinaryOp::Div()
        | BinaryOp::Modulo() => (mk_uniftype::num(), mk_uniftype::num(), mk_uniftype::num()),
        // Sym -> Dyn -> Dyn -> Dyn
        BinaryOp::Seal() => (
            mk_uniftype::sym(),
            mk_uniftype::dynamic(),
            mk_uty_arrow!(TypeF::Dyn, TypeF::Dyn),
        ),
        // Str -> Str -> Str
        BinaryOp::StrConcat() => (mk_uniftype::str(), mk_uniftype::str(), mk_uniftype::str()),
        // Ideally: Contract -> Label -> Dyn -> Dyn
        // Currently: Dyn -> Dyn -> (Dyn -> Dyn)
        BinaryOp::Assume() => (
            mk_uniftype::dynamic(),
            mk_uniftype::dynamic(),
            mk_uty_arrow!(mk_uniftype::dynamic(), mk_uniftype::dynamic()),
        ),
        // Sym -> Dyn -> Dyn -> Dyn
        BinaryOp::Unseal() => (
            mk_uniftype::sym(),
            mk_uniftype::dynamic(),
            mk_uty_arrow!(TypeF::Dyn, TypeF::Dyn),
        ),
        // forall a b. a -> b -> Bool
        BinaryOp::Eq() => (
            UnifType::UnifVar(state.table.fresh_type_var_id()),
            UnifType::UnifVar(state.table.fresh_type_var_id()),
            mk_uniftype::bool(),
        ),
        // Num -> Num -> Bool
        BinaryOp::LessThan()
        | BinaryOp::LessOrEq()
        | BinaryOp::GreaterThan()
        | BinaryOp::GreaterOrEq() => (mk_uniftype::num(), mk_uniftype::num(), mk_uniftype::bool()),
        // Str -> Dyn -> Dyn
        BinaryOp::GoField() => (
            mk_uniftype::str(),
            mk_uniftype::dynamic(),
            mk_uniftype::dynamic(),
        ),
        // forall a. Str -> { _ : a} -> a
        BinaryOp::DynAccess() => {
            let res = UnifType::UnifVar(state.table.fresh_type_var_id());

            (mk_uniftype::str(), mk_uniftype::dict(res.clone()), res)
        }
        // forall a. Str -> {_ : a} -> a -> {_ : a}
        BinaryOp::DynExtend {
            ext_kind: RecordExtKind::WithValue,
            ..
        } => {
            let res = UnifType::UnifVar(state.table.fresh_type_var_id());
            (
                mk_uniftype::str(),
                mk_uniftype::dict(res.clone()),
                mk_uty_arrow!(res.clone(), mk_uniftype::dict(res)),
            )
        }
        // forall a. Str -> {_ : a} -> {_ : a}
        BinaryOp::DynExtend {
            ext_kind: RecordExtKind::WithoutValue,
            ..
        } => {
            let res = UnifType::UnifVar(state.table.fresh_type_var_id());
            (
                mk_uniftype::str(),
                mk_uniftype::dict(res.clone()),
                mk_uty_arrow!(res.clone(), mk_uniftype::dict(res)),
            )
        }
        // forall a. Str -> { _ : a } -> { _ : a}
        BinaryOp::DynRemove() => {
            let res = UnifType::UnifVar(state.table.fresh_type_var_id());
            (
                mk_uniftype::str(),
                mk_uniftype::dict(res.clone()),
                mk_uniftype::dict(res),
            )
        }
        // forall a. Str -> {_: a} -> Bool
        BinaryOp::HasField() => {
            let ty_elt = UnifType::UnifVar(state.table.fresh_type_var_id());
            (
                mk_uniftype::str(),
                mk_uniftype::dict(ty_elt),
                mk_uniftype::bool(),
            )
        }
        // forall a. Array a -> Array a -> Array a
        BinaryOp::ArrayConcat() => {
            let ty_elt = UnifType::UnifVar(state.table.fresh_type_var_id());
            let ty_array = mk_uniftype::array(ty_elt);
            (ty_array.clone(), ty_array.clone(), ty_array)
        }
        // forall a. Array a -> Num -> a
        BinaryOp::ArrayElemAt() => {
            let ty_elt = UnifType::UnifVar(state.table.fresh_type_var_id());
            (
                mk_uniftype::array(ty_elt.clone()),
                mk_uniftype::num(),
                ty_elt,
            )
        }
        // Dyn -> Dyn -> Dyn
        BinaryOp::Merge(_) => (
            mk_uniftype::dynamic(),
            mk_uniftype::dynamic(),
            mk_uniftype::dynamic(),
        ),
        // <Md5, Sha1, Sha256, Sha512> -> Str -> Str
        BinaryOp::Hash() => (
            mk_uty_enum!("Md5", "Sha1", "Sha256", "Sha512"),
            mk_uniftype::str(),
            mk_uniftype::str(),
        ),
        // forall a. <Json, Yaml, Toml> -> a -> Str
        BinaryOp::Serialize() => {
            let ty_input = UnifType::UnifVar(state.table.fresh_type_var_id());
            (
                mk_uty_enum!("Json", "Yaml", "Toml"),
                ty_input,
                mk_uniftype::str(),
            )
        }
        // <Json, Yaml, Toml> -> Str -> Dyn
        BinaryOp::Deserialize() => (
            mk_uty_enum!("Json", "Yaml", "Toml"),
            mk_uniftype::str(),
            mk_uniftype::dynamic(),
        ),
        // Num -> Num -> Num
        BinaryOp::Pow() => (mk_uniftype::num(), mk_uniftype::num(), mk_uniftype::num()),
        // Str -> Str -> Bool
        BinaryOp::StrContains() => (mk_uniftype::str(), mk_uniftype::str(), mk_uniftype::bool()),
        // Str -> Str -> Array Str
        BinaryOp::StrSplit() => (
            mk_uniftype::str(),
            mk_uniftype::str(),
            mk_uniftype::array(TypeF::String),
        ),
        // The first argument is a contract, the second is a label.
        // forall a. Dyn -> Dyn -> Array a -> Array a
        BinaryOp::ArrayLazyAssume() => {
            let ty_elt = UnifType::UnifVar(state.table.fresh_type_var_id());
            let ty_array = mk_uniftype::array(ty_elt);
            (
                mk_uniftype::dynamic(),
                mk_uniftype::dynamic(),
                mk_uty_arrow!(ty_array.clone(), ty_array),
            )
        }
        // The first argument is a label, the third is a contract.
        // forall a. Dyn -> {_: a} -> Dyn -> {_: a}
        BinaryOp::RecordLazyAssume() => {
            let ty_field = UnifType::UnifVar(state.table.fresh_type_var_id());
            let ty_dict = mk_uniftype::dict(ty_field);
            (
                mk_uniftype::dynamic(),
                ty_dict.clone(),
                mk_uty_arrow!(mk_uniftype::dynamic(), ty_dict),
            )
        }
        // Morally: Str -> Lbl -> Lbl
        // Actual: Str -> Dyn -> Dyn
        BinaryOp::LabelWithMessage() => (
            mk_uniftype::str(),
            mk_uniftype::dynamic(),
            mk_uniftype::dynamic(),
        ),
        // Morally: Array Str -> Lbl -> Lbl
        // Actual: Array Str -> Dyn -> Dyn
        BinaryOp::LabelWithNotes() => (
            mk_uniftype::array(TypeF::String),
            mk_uniftype::dynamic(),
            mk_uniftype::dynamic(),
        ),
        // Morally: Str -> Lbl -> Lbl
        // Actual: Str -> Dyn -> Dyn
        BinaryOp::LabelAppendNote() => (
            mk_uniftype::str(),
            mk_uniftype::dynamic(),
            mk_uniftype::dynamic(),
        ),
        // Morally: Sym -> Lbl -> TypeVarData
        // Actual: Sym -> Dyn -> TypeVarData
        BinaryOp::LookupTypeVar() => (
            mk_uniftype::sym(),
            mk_uniftype::dynamic(),
            TypeVarData::unif_type(),
        ),
    })
}

pub fn get_nop_type(
    state: &mut State,
    op: &NAryOp,
) -> Result<(Vec<UnifType>, UnifType), TypecheckError> {
    Ok(match op {
        // Str -> Str -> Str -> Str
        NAryOp::StrReplace() | NAryOp::StrReplaceRegex() => (
            vec![mk_uniftype::str(), mk_uniftype::str(), mk_uniftype::str()],
            mk_uniftype::str(),
        ),
        // Str -> Num -> Num -> Str
        NAryOp::StrSubstr() => (
            vec![mk_uniftype::str(), mk_uniftype::num(), mk_uniftype::num()],
            mk_uniftype::str(),
        ),
        // Dyn -> Dyn -> Dyn -> Dyn -> Dyn
        NAryOp::RecordSealTail() => (
            vec![
                mk_uniftype::dynamic(),
                mk_uniftype::dynamic(),
                mk_uniftype::dict(mk_uniftype::dynamic()),
                mk_uniftype::dict(mk_uniftype::dynamic()),
            ],
            mk_uniftype::dynamic(),
        ),
        // Dyn -> Dyn -> Dyn -> Dyn
        NAryOp::RecordUnsealTail() => (
            vec![
                mk_uniftype::dynamic(),
                mk_uniftype::dynamic(),
                mk_uniftype::dict(mk_uniftype::dynamic()),
            ],
            mk_uniftype::dynamic(),
        ),
        // Num -> Num -> Array a -> Array a
        NAryOp::ArraySlice() => {
            let element_type = state.table.fresh_type_uvar();

            (
                vec![
                    mk_uniftype::num(),
                    mk_uniftype::num(),
                    mk_uniftype::array(element_type.clone()),
                ],
                mk_uniftype::array(element_type),
            )
        }
        // This should not happen, as MergeContract() is only produced during evaluation.
        NAryOp::MergeContract() => panic!("cannot typecheck MergeContract()"),
        // Morally: Sym -> Polarity -> Lbl -> Lbl
        // Actual: Sym -> Polarity -> Dyn -> Dyn
        NAryOp::InsertTypeVar() => (
            vec![
                mk_uniftype::sym(),
                Polarity::unif_type(),
                mk_uniftype::dynamic(),
            ],
            mk_uniftype::dynamic(),
        ),
    })
}
