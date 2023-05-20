//! Serialization of an evaluated program to various data format.
use malachite::{num::conversion::traits::RoundingFrom, rounding_modes::RoundingMode};

use crate::{
    error::ExportError,
    term::{
        array::{Array, ArrayAttrs},
        record::RecordData,
        IndexMap, Number, RichTerm, Term, TypeAnnotation,
    },
};

use serde::{
    de::{Deserialize, Deserializer},
    ser::{Serialize, SerializeMap, SerializeSeq, Serializer},
};

use malachite::num::conversion::traits::IsInteger;

use std::{fmt, io, rc::Rc, str::FromStr};

/// Available export formats.
// If you add or remove variants, remember to update the CLI docs in `src/bin/nickel.rs'
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub enum ExportFormat {
    Raw,
    #[default]
    Json,
    Yaml,
    Toml,
}

impl fmt::Display for ExportFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Raw => write!(f, "raw"),
            Self::Json => write!(f, "json"),
            Self::Yaml => write!(f, "yaml"),
            Self::Toml => write!(f, "toml"),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct ParseFormatError(String);

impl fmt::Display for ParseFormatError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "unsupported export format {}", self.0)
    }
}

impl FromStr for ExportFormat {
    type Err = ParseFormatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_ref() {
            "raw" => Ok(ExportFormat::Raw),
            "json" => Ok(ExportFormat::Json),
            "yaml" => Ok(ExportFormat::Yaml),
            "toml" => Ok(ExportFormat::Toml),
            _ => Err(ParseFormatError(String::from(s))),
        }
    }
}

/// Implicitly convert numbers to primitive integers when possible, and serialize an exact
/// representation. Note that `u128` and `i128` aren't supported for common configuration formats
/// in serde, so we rather pick `i64` and `u64`, even if the former couple theoretically allows for
/// a wider range of rationals to be exactly represented. We don't expect values to be that large
/// in practice anyway: using arbitrary precision rationals is directed toward not introducing rounding errors
/// when performing simple arithmetic operations over decimals numbers, mostly.
///
/// If the number doesn't fit into an `i64` or `u64`, we approximate it by the nearest `f64` and
/// serialize this value. This may incur a loss of precision, but this is expected: we can't
/// represent something like e.g. `1/3` exactly in JSON anyway.
pub fn serialize_num<S>(n: &Number, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    if n.is_integer() {
        if *n < 0 {
            if let Ok(n_as_integer) = i64::try_from(n) {
                return n_as_integer.serialize(serializer);
            }
        } else if let Ok(n_as_uinteger) = u64::try_from(n) {
            return n_as_uinteger.serialize(serializer);
        }
    }

    f64::rounding_from(n, RoundingMode::Nearest).serialize(serializer)
}

/// Deserialize a Nickel number. As for parsing, we convert the number from a 64bits float
pub fn deserialize_num<'de, D>(deserializer: D) -> Result<Number, D::Error>
where
    D: Deserializer<'de>,
{
    let as_f64 = f64::deserialize(deserializer)?;
    Number::try_from_float_simplest(as_f64).map_err(|_| {
        serde::de::Error::custom(format!(
            "couldn't convert {as_f64} to a Nickel number: Nickel doesn't support NaN nor infinity"
        ))
    })
}

/// Serializer for annotated values.
pub fn serialize_annotated_value<S>(
    _annot: &TypeAnnotation,
    t: &RichTerm,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    t.serialize(serializer)
}

static RENAME_RULES: &[(&str, RenameRule)] = &[
    ("lowercase", RenameRule::LowerCase),
    ("UPPERCASE", RenameRule::UpperCase),
    ("PascalCase", RenameRule::PascalCase),
    ("camelCase", RenameRule::CamelCase),
    ("snake_case", RenameRule::SnakeCase),
    ("SCREAMING_SNAKE_CASE", RenameRule::ScreamingSnakeCase),
    ("kebab-case", RenameRule::KebabCase),
    ("SCREAMING-KEBAB-CASE", RenameRule::ScreamingKebabCase),
];

impl FromStr for RenameRule {
    type Err = ParseFormatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        for (name, rule) in RENAME_RULES {
            if s == *name {
                return Ok(*rule);
            }
        }

        Err(ParseFormatError(s.into()))
    }
}

/// The different possible ways to change case of fields in a struct, or variants in an enum.
#[derive(Debug, Default, Copy, Clone, PartialEq)]
pub enum RenameRule {
    /// Don't apply a default rename rule.
    #[default]
    None,
    /// Rename direct children to "lowercase" style.
    LowerCase,
    /// Rename direct children to "UPPERCASE" style.
    UpperCase,
    /// Rename direct children to "PascalCase" style, as typically used for
    /// enum variants.
    PascalCase,
    /// Rename direct children to "camelCase" style.
    CamelCase,
    /// Rename direct children to "snake_case" style, as commonly used for
    /// fields.
    SnakeCase,
    /// Rename direct children to "SCREAMING_SNAKE_CASE" style, as commonly
    /// used for constants.
    ScreamingSnakeCase,
    /// Rename direct children to "kebab-case" style.
    KebabCase,
    /// Rename direct children to "SCREAMING-KEBAB-CASE" style.
    ScreamingKebabCase,
}

impl RenameRule {
    /// Apply a renaming rule to an enum variant, returning the version expected in the source.
    pub fn apply_to_variant(&self, variant: &str) -> String {
        use self::RenameRule::*;

        match *self {
            None | PascalCase => variant.to_owned(),
            LowerCase => variant.to_ascii_lowercase(),
            UpperCase => variant.to_ascii_uppercase(),
            CamelCase => variant[..1].to_ascii_lowercase() + &variant[1..],
            SnakeCase => {
                let mut snake = String::new();
                for (i, ch) in variant.char_indices() {
                    if i > 0 && ch.is_uppercase() {
                        snake.push('_');
                    }
                    snake.push(ch.to_ascii_lowercase());
                }
                snake
            }
            ScreamingSnakeCase => SnakeCase.apply_to_variant(variant).to_ascii_uppercase(),
            KebabCase => SnakeCase.apply_to_variant(variant).replace('_', "-"),
            ScreamingKebabCase => ScreamingSnakeCase
                .apply_to_variant(variant)
                .replace('_', "-"),
        }
    }
}

/// Serializer for a record. Serialize fields in alphabetical order to get a deterministic output
pub fn serialize_record<S>(record: &RecordData, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut entries = record
        .iter_serializable()
        .collect::<Result<Vec<_>, _>>()
        .map_err(|missing_def_err| {
            serde::ser::Error::custom(format!(
                "missing field definition for `{}`",
                missing_def_err.id
            ))
        })?;

    entries.sort_by_key(|(k, _)| *k);

    let mut map_ser = serializer.serialize_map(Some(entries.len()))?;
    for (id, t) in entries.iter() {
        map_ser.serialize_entry(&id.to_string(), &t)?
    }

    map_ser.end()
}

/// Deserialize for a record. Required to set the record attributes to default.
pub fn deserialize_record<'de, D>(deserializer: D) -> Result<RecordData, D::Error>
where
    D: Deserializer<'de>,
{
    let fields = IndexMap::deserialize(deserializer)?;
    Ok(RecordData::with_field_values(fields))
}

/// Serialize for an Array. Required to hide the internal attributes.
pub fn serialize_array<S>(
    terms: &Array,
    _attrs: &ArrayAttrs,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    let mut seq = serializer.serialize_seq(Some(terms.len()))?;
    for term in terms.iter() {
        seq.serialize_element(term)?;
    }

    seq.end()
}

/// Deserialize for an Array. Required to set the default attributes.
pub fn deserialize_array<'de, D>(deserializer: D) -> Result<(Array, ArrayAttrs), D::Error>
where
    D: Deserializer<'de>,
{
    let terms = Array::new(Rc::from(Vec::deserialize(deserializer)?));
    Ok((terms, Default::default()))
}

impl Serialize for RichTerm {
    /// Serialize the underlying term.
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (*self.term).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for RichTerm {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let t: Term = Term::deserialize(deserializer)?;
        Ok(RichTerm::from(t))
    }
}

/// Check that a term is serializable. Serializable terms are booleans, numbers, strings, enum,
/// arrays of serializable terms or records of serializable terms.
pub fn validate(format: ExportFormat, t: &RichTerm) -> Result<(), ExportError> {
    use Term::*;

    if format == ExportFormat::Raw {
        if let Term::Str(_) = t.term.as_ref() {
            Ok(())
        } else {
            Err(ExportError::NotAString(t.clone()))
        }
    } else {
        match t.term.as_ref() {
            // TOML doesn't support null values
            Null if format == ExportFormat::Json || format == ExportFormat::Yaml => Ok(()),
            Null => Err(ExportError::UnsupportedNull(format, t.clone())),
            Bool(_) | Num(_) | Str(_) | Enum(_) => Ok(()),
            Record(record) => {
                record.iter_serializable().try_for_each(|binding| {
                    // unwrap(): terms must be fully evaluated before being validated for
                    // serialization. Otherwise, it's an internal error.
                    let (_, rt) = binding.unwrap_or_else(|err| panic!("encountered field without definition `{}` during pre-serialization validation", err.id));
                    validate(format, rt)
                })?;
                Ok(())
            }
            Array(array, _) => {
                array.iter().try_for_each(|t| validate(format, t))?;
                Ok(())
            }
            _ => Err(ExportError::NonSerializable(t.clone())),
        }
    }
}

macro_rules! forward_ser {
    ($fn:ident, $typ:ty) => {
        fn $fn(self, v: $typ) -> Result<Self::Ok, Self::Error> {
            self.0.$fn(v)
        }
    };
}

macro_rules! forward_ser_todo {
    ($fn:ident, $typ:ty) => {
        fn $fn(self, v: $typ) -> Result<Self::Ok, Self::Error> {
            todo!()
        }
    };
}

struct RenamingMapSerializer<S>(S);
impl<S: serde::ser::SerializeMap> serde::ser::SerializeMap for RenamingMapSerializer<S> {
    type Ok = S::Ok;
    type Error = S::Error;

    fn serialize_entry<K: ?Sized, V: ?Sized>(
            &mut self,
            key: &K,
            value: &V,
        ) -> Result<(), Self::Error>
        where
            K: Serialize,
            V: Serialize, {
        self.0.serialize_entry(
            &key.serialize(KeyRenamerSerializer).unwrap(),
            value)
    }

    fn serialize_key<T: ?Sized>(&mut self, key: &T) -> Result<(), Self::Error>
        where
            T: Serialize {
        self.0.serialize_key(&key.serialize(KeyRenamerSerializer).unwrap())
    }

    fn serialize_value<T: ?Sized>(&mut self, value: &T) -> Result<(), Self::Error>
        where
            T: Serialize {
        self.0.serialize_value(value)
    }

    fn end(self) -> Result<Self::Ok, Self::Error> {
        self.0.end()
    }
}

struct KeyRenamerSerializer;

impl Serializer for KeyRenamerSerializer {
    type SerializeSeq = serde::ser::Impossible<Self::Ok, toml::ser::Error>;
    type SerializeTuple = serde::ser::Impossible<Self::Ok, toml::ser::Error>;
    type SerializeTupleStruct = serde::ser::Impossible<Self::Ok, toml::ser::Error>;
    type SerializeTupleVariant = serde::ser::Impossible<Self::Ok, toml::ser::Error>;
    type SerializeMap = serde::ser::Impossible<Self::Ok, toml::ser::Error>;
    type SerializeStruct = serde::ser::Impossible<Self::Ok, toml::ser::Error>;
    type SerializeStructVariant = serde::ser::Impossible<Self::Ok, toml::ser::Error>;

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        todo!()
    }

    // Now, if that one function above would be all the magic you need, it's be nice
    // But it turns out there are a lot more places where you can change the case of the name
    // Do you want to implement them all? I'll pass

    fn serialize_unit_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_newtype_struct<T: ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        todo!()
    }

    fn serialize_newtype_variant<T: ?Sized>(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        todo!()
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        todo!()
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        todo!()
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        todo!()
    }

    fn serialize_str(self, value: &str) -> Result<String, Self::Error> {
        Ok(RenameRule::KebabCase.apply_to_variant(value))
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        todo!()
    }

    // Need a bunch of boilderplate, too.

    type Ok = String;
    type Error = toml::ser::Error;

    forward_ser_todo! {serialize_bool, bool}
    forward_ser_todo! {serialize_i8, i8}
    forward_ser_todo! {serialize_i16, i16}
    forward_ser_todo! {serialize_i32, i32}
    forward_ser_todo! {serialize_i64, i64}
    forward_ser_todo! {serialize_u8, u8}
    forward_ser_todo! {serialize_u16, u16}
    forward_ser_todo! {serialize_u32, u32}
    forward_ser_todo! {serialize_u64, u64}
    forward_ser_todo! {serialize_f32, f32}
    forward_ser_todo! {serialize_f64, f64}
    forward_ser_todo! {serialize_char, char}
    forward_ser_todo! {serialize_bytes, &[u8]}
    forward_ser_todo! {serialize_unit_struct, &'static str}

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_some<T: ?Sized>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        todo!()
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        todo!()
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        todo!()
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        todo!()
    }
}

struct RenamingSerializer<S>(S);

impl<S: Serializer> Serializer for RenamingSerializer<S> {
    // The magic first: redirect the struct serializer to one that
    type SerializeStruct = S::SerializeStruct;

    fn serialize_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStruct, Self::Error> {
        self.0.serialize_struct(name, len)
    }

    // Now, if that one function above would be all the magic you need, it's be nice
    // But it turns out there are a lot more places where you can change the case of the name
    // Do you want to implement them all? I'll pass

    fn serialize_unit_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
    ) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit_variant(name, variant_index, variant)
    }

    fn serialize_newtype_struct<T: ?Sized>(
        self,
        name: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        self.0.serialize_newtype_struct(name, value)
    }

    fn serialize_newtype_variant<T: ?Sized>(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        self.0.serialize_newtype_variant(name, variant_index, variant, value)
    }

    fn serialize_tuple_struct(
        self,
        name: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleStruct, Self::Error> {
        self.0.serialize_tuple_struct(name, len)
    }

    fn serialize_tuple_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeTupleVariant, Self::Error> {
        self.0.serialize_tuple_variant(name, variant_index, variant, len)
    }

    fn serialize_map(self, len: Option<usize>) -> Result<Self::SerializeMap, Self::Error> {
        self.0.serialize_map(len).map(|s| RenamingMapSerializer(s))
    }

    fn serialize_struct_variant(
        self,
        name: &'static str,
        variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<Self::SerializeStructVariant, Self::Error> {
        self.0.serialize_struct_variant(name, variant_index, variant, len)
    }

    type SerializeSeq = S::SerializeSeq;
    type SerializeTuple = S::SerializeTuple;
    type SerializeTupleStruct = S::SerializeTupleStruct;
    type SerializeTupleVariant = S::SerializeTupleVariant;
    type SerializeMap = RenamingMapSerializer<S::SerializeMap>;
    type SerializeStructVariant = S::SerializeStructVariant;

    // Need a bunch of boilderplate, too.

    type Ok = S::Ok;
    type Error = S::Error;

    forward_ser! {serialize_bool, bool}
    forward_ser! {serialize_i8, i8}
    forward_ser! {serialize_i16, i16}
    forward_ser! {serialize_i32, i32}
    forward_ser! {serialize_i64, i64}
    forward_ser! {serialize_u8, u8}
    forward_ser! {serialize_u16, u16}
    forward_ser! {serialize_u32, u32}
    forward_ser! {serialize_u64, u64}
    forward_ser! {serialize_f32, f32}
    forward_ser! {serialize_f64, f64}
    forward_ser! {serialize_char, char}
    forward_ser! {serialize_str, &str}
    forward_ser! {serialize_bytes, &[u8]}
    forward_ser! {serialize_unit_struct, &'static str}

    fn serialize_none(self) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_none()
    }

    fn serialize_some<T: ?Sized>(self, value: &T) -> Result<Self::Ok, Self::Error>
    where
        T: Serialize,
    {
        self.0.serialize_some(value)
    }

    fn serialize_unit(self) -> Result<Self::Ok, Self::Error> {
        self.0.serialize_unit()
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<Self::SerializeSeq, Self::Error> {
        self.0.serialize_seq(len)
    }

    fn serialize_tuple(self, len: usize) -> Result<Self::SerializeTuple, Self::Error> {
        self.0.serialize_tuple(len)
    }
}

pub fn to_writer<W>(
    mut writer: W,
    format: ExportFormat,
    rename_rule: RenameRule,
    rt: &RichTerm,
) -> Result<(), ExportError>
where
    W: io::Write,
{
    match format {
        ExportFormat::Json => serde_json::to_writer_pretty(writer, &rt)
            .map_err(|err| ExportError::Other(err.to_string())),
        ExportFormat::Yaml => {
            serde_yaml::to_writer(writer, &rt).map_err(|err| ExportError::Other(err.to_string()))
        }
        ExportFormat::Toml => {
            let mut value = String::new();
            let s = RenamingSerializer(toml::Serializer::pretty(&mut value));

            serde::Serialize::serialize(&rt, s)
            .map_err(|err| ExportError::Other(err.to_string()))
            .and_then(|_s| {
                writer
                    .write_all(value.as_bytes())
                    .map_err(|err| ExportError::Other(err.to_string()))
            })
        }
        ExportFormat::Raw => match rt.as_ref() {
            Term::Str(s) => writer
                .write_all(s.as_bytes())
                .map_err(|err| ExportError::Other(err.to_string())),
            t => Err(ExportError::Other(format!(
                "raw export requires a `String`, got {}",
                // unwrap(): terms must be fully evaluated before serialization,
                // and fully evaluated terms have a definite type.
                t.type_of().unwrap()
            ))),
        },
    }
}

pub fn to_string(format: ExportFormat, rt: &RichTerm) -> Result<String, ExportError> {
    let mut buffer: Vec<u8> = Vec::new();
    to_writer(&mut buffer, format, RenameRule::default(), rt)?;

    Ok(String::from_utf8_lossy(&buffer).into_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::resolvers::DummyResolver;
    use crate::error::{Error, EvalError};
    use crate::eval::cache::CacheImpl;
    use crate::eval::{Environment, VirtualMachine};
    use crate::position::TermPos;
    use crate::program::Program;
    use crate::term::{make as mk_term, BinaryOp};
    use serde_json::json;
    use std::io::Cursor;

    fn mk_program(s: &str) -> Result<Program<CacheImpl>, Error> {
        let src = Cursor::new(s);

        Program::new_from_source(src, "<test>").map_err(|io_err| {
            Error::EvalError(EvalError::Other(
                format!("IO error: {io_err}"),
                TermPos::None,
            ))
        })
    }

    macro_rules! assert_json_eq {
        ( $term:expr, $result:expr ) => {
            assert_eq!(
                serde_json::to_string(&mk_program($term).and_then(|mut p| p.eval_full()).unwrap())
                    .unwrap(),
                serde_json::to_string(&$result).unwrap()
            )
        };
    }

    macro_rules! assert_pass_validation {
        ( $format:expr, $term:expr, true) => {
            validate(
                $format,
                &mk_program($term)
                    .and_then(|mut p| p.eval_full())
                    .unwrap()
                    .into(),
            )
            .unwrap();
        };
        ( $format:expr, $term:expr, false) => {
            validate(
                $format,
                &mk_program($term)
                    .and_then(|mut p| p.eval_full())
                    .unwrap()
                    .into(),
            )
            .unwrap_err();
        };
    }

    macro_rules! assert_involutory {
        ( $term:expr ) => {
            let evaluated = mk_program($term).and_then(|mut p| p.eval_full()).unwrap();
            let from_json: RichTerm =
                serde_json::from_str(&serde_json::to_string(&evaluated).unwrap()).unwrap();
            let from_yaml: RichTerm =
                serde_yaml::from_str(&serde_yaml::to_string(&evaluated).unwrap()).unwrap();
            let from_toml: RichTerm =
                toml::from_str(&format!("{}", &toml::to_string(&evaluated).unwrap())).unwrap();

            assert_eq!(
                VirtualMachine::<_, CacheImpl>::new(DummyResolver {})
                    .eval(
                        mk_term::op2(BinaryOp::Eq(), from_json, evaluated.clone()),
                        &Environment::new(),
                    )
                    .map(Term::from),
                Ok(Term::Bool(true))
            );
            assert_eq!(
                VirtualMachine::<_, CacheImpl>::new(DummyResolver {})
                    .eval(
                        mk_term::op2(BinaryOp::Eq(), from_yaml, evaluated.clone()),
                        &Environment::new(),
                    )
                    .map(Term::from),
                Ok(Term::Bool(true))
            );
            assert_eq!(
                VirtualMachine::<_, CacheImpl>::new(DummyResolver {})
                    .eval(
                        mk_term::op2(BinaryOp::Eq(), from_toml, evaluated),
                        &Environment::new(),
                    )
                    .map(Term::from),
                Ok(Term::Bool(true))
            );
        };
    }

    #[test]
    fn basic() {
        assert_json_eq!("1 + 1", 2);

        let null: Option<()> = None;
        assert_json_eq!("null", null);

        assert_json_eq!("if true then false else true", false);
        assert_json_eq!(r##""Hello, %{"world"}!""##, "Hello, world!");
        assert_json_eq!("'foo", "foo");
    }

    #[test]
    fn arrays() {
        assert_json_eq!("[]", json!([]));
        assert_json_eq!("[null, (1+1), (2+2), (3+3)]", json!([null, 2, 4, 6]));
        assert_json_eq!(
            r##"['a, ("b" ++ "c"), "d%{"e"}f", "g"]"##,
            json!(["a", "bc", "def", "g"])
        );
        assert_json_eq!(
            r#"std.array.fold_right (fun elt acc => [[elt]] @ acc) [] [1, 2, 3, 4]"#,
            json!([[1], [2], [3], [4]])
        );
        assert_json_eq!("[\"a\", 1, false, 'foo]", json!(["a", 1, false, "foo"]));
    }

    #[test]
    fn records() {
        assert_json_eq!(
            "{a = 1, b = 2+2, c = 3, d = null}",
            json!({"a": 1, "b": 4, "c": 3, "d": null})
        );

        assert_json_eq!(
            "{a = {} & {b = {c = if true then 'richtig else 'falsch}}}",
            json!({"a": {"b": {"c": "richtig"}}})
        );

        assert_json_eq!(
            "{foo = let z = 0.5 + 0.5 in z, bar = [\"str\", true || false], baz = {subfoo = !false} & {subbar = 1 - 1}}",
            json!({"foo": 1, "bar": ["str", true], "baz": {"subfoo": true, "subbar": 0}})
        );
    }

    #[test]
    fn meta_values() {
        assert_json_eq!(
            "{a | default = 1, b | doc \"doc\" = 2+2, c = 3}",
            json!({"a": 1, "b": 4, "c": 3})
        );

        assert_json_eq!(
            "{a = {b | default = {}} & {b.c | default = (if true then 'faux else 'vrai)}}",
            json!({"a": {"b": {"c": "faux"}}})
        );

        assert_json_eq!(
            "{baz | default = {subfoo | default = !false} & {subbar | default = 1 - 1}}",
            json!({"baz": {"subfoo": true, "subbar": 0}})
        );

        assert_json_eq!(
            "{a = {b | default = {}} & {b.c | not_exported = false} & {b.d = true}}",
            json!({"a": {"b": {"d": true}}})
        );
    }

    #[test]
    fn prevalidation() {
        assert_pass_validation!(ExportFormat::Json, "{a = 1, b = {c = fun x => x}}", false);
        assert_pass_validation!(
            ExportFormat::Json,
            "{foo.bar = let y = \"a\" in y, b = [[fun x => x]]}",
            false
        );
        assert_pass_validation!(ExportFormat::Json, "{foo = null}", true);
        assert_pass_validation!(ExportFormat::Toml, "{foo = null}", false);
    }

    #[test]
    fn involution() {
        assert_involutory!("{val = 1 + 1}");
        assert_involutory!("{val = \"Some string\"}");
        assert_involutory!("{val = [\"a\", 3, []]}");
        assert_involutory!("{a.foo.bar = \"2\", b = false, c = [{d = \"e\"}, {d = \"f\"}]}");
    }
}
