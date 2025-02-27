use super::cache::CacheImpl;
use super::*;
use crate::cache::resolvers::{DummyResolver, SimpleResolver};
use crate::error::ImportError;
use crate::label::Label;
use crate::parser::{grammar, lexer, ErrorTolerantParser};
use crate::term::make as mk_term;
use crate::term::Number;
use crate::term::{BinaryOp, StrChunk, UnaryOp};
use crate::transform::import_resolution::strict::resolve_imports;
use crate::{mk_app, mk_fun};
use codespan::Files;

/// Evaluate a term without import support.
fn eval_no_import(t: RichTerm) -> Result<Term, EvalError> {
    VirtualMachine::<_, CacheImpl>::new(DummyResolver {})
        .eval(t, &Environment::new())
        .map(Term::from)
}

fn parse(s: &str) -> Option<RichTerm> {
    let id = Files::new().add("<test>", String::from(s));

    grammar::TermParser::new()
        .parse_strict(id, lexer::Lexer::new(s))
        .map(RichTerm::without_pos)
        .map_err(|err| println!("{err:?}"))
        .ok()
}

#[test]
fn identity_over_values() {
    let num = Term::Num(Number::try_from(45.3).unwrap());
    assert_eq!(Ok(num.clone()), eval_no_import(num.into()));

    let boolean = Term::Bool(true);
    assert_eq!(Ok(boolean.clone()), eval_no_import(boolean.into()));

    let lambda = mk_fun!("x", mk_app!(mk_term::var("x"), mk_term::var("x")));
    assert_eq!(Ok(lambda.as_ref().clone()), eval_no_import(lambda));
}

#[test]
fn blame_panics() {
    let l = Label::dummy();
    if let Err(EvalError::BlameError {
        evaluated_arg: _,
        label,
        call_stack: _,
    }) = eval_no_import(mk_term::op1(UnaryOp::Blame(), Term::Lbl(l.clone())))
    {
        assert_eq!(label, l);
    } else {
        panic!("This evaluation should've returned a BlameError!");
    }
}

#[test]
#[should_panic]
fn lone_var_panics() {
    eval_no_import(mk_term::var("unbound")).unwrap();
}

#[test]
fn only_fun_are_applicable() {
    eval_no_import(mk_app!(Term::Bool(true), mk_term::integer(45))).unwrap_err();
}

#[test]
fn simple_app() {
    let t = mk_app!(mk_term::id(), mk_term::integer(5));
    assert_eq!(Ok(Term::Num(Number::from(5))), eval_no_import(t));
}

#[test]
fn simple_let() {
    let t = mk_term::let_in("x", mk_term::integer(5), mk_term::var("x"));
    assert_eq!(Ok(Term::Num(Number::from(5))), eval_no_import(t));
}

#[test]
fn simple_ite() {
    let t = mk_term::if_then_else(Term::Bool(true), mk_term::integer(5), Term::Bool(false));
    assert_eq!(Ok(Term::Num(Number::from(5))), eval_no_import(t));
}

#[test]
fn simple_plus() {
    let t = mk_term::op2(
        BinaryOp::Plus(),
        mk_term::integer(5),
        Term::Num(Number::try_from(7.5).unwrap()),
    );
    assert_eq!(
        Ok(Term::Num(Number::try_from(12.5).unwrap())),
        eval_no_import(t)
    );
}

#[test]
fn asking_for_various_types() {
    let num = mk_term::op1(
        UnaryOp::Typeof(),
        Term::Num(Number::try_from(45.3).unwrap()),
    );
    assert_eq!(Ok(Term::Enum("Number".into())), eval_no_import(num));

    let boolean = mk_term::op1(UnaryOp::Typeof(), Term::Bool(true));
    assert_eq!(Ok(Term::Enum("Bool".into())), eval_no_import(boolean));

    let lambda = mk_term::op1(
        UnaryOp::Typeof(),
        mk_fun!("x", mk_app!(mk_term::var("x"), mk_term::var("x"))),
    );
    assert_eq!(Ok(Term::Enum("Function".into())), eval_no_import(lambda));
}

#[test]
fn imports() {
    let mut vm = VirtualMachine::new(SimpleResolver::new());
    vm.import_resolver_mut()
        .add_source(String::from("two"), String::from("1 + 1"));
    vm.import_resolver_mut()
        .add_source(String::from("lib"), String::from("{f = true}"));
    vm.import_resolver_mut()
        .add_source(String::from("bad"), String::from("^$*/.23ab 0°@"));
    vm.import_resolver_mut().add_source(
        String::from("nested"),
        String::from("let x = import \"two\" in x + 1"),
    );
    vm.import_resolver_mut().add_source(
        String::from("cycle"),
        String::from("let x = import \"cycle_b\" in {a = 1, b = x.a}"),
    );
    vm.import_resolver_mut().add_source(
        String::from("cycle_b"),
        String::from("let x = import \"cycle\" in {a = x.a}"),
    );

    fn mk_import<R>(
        var: &str,
        import: &str,
        body: RichTerm,
        vm: &mut VirtualMachine<R, CacheImpl>,
    ) -> Result<RichTerm, ImportError>
    where
        R: ImportResolver,
    {
        resolve_imports(
            mk_term::let_in(var, mk_term::import(import), body),
            vm.import_resolver_mut(),
        )
        .map(|resolve_result| resolve_result.transformed_term)
    }

    // let x = import "does_not_exist" in x
    match mk_import("x", "does_not_exist", mk_term::var("x"), &mut vm).unwrap_err() {
        ImportError::IOError(_, _, _) => (),
        _ => panic!(),
    };

    // let x = import "bad" in x
    match mk_import("x", "bad", mk_term::var("x"), &mut vm).unwrap_err() {
        ImportError::ParseErrors(_, _) => (),
        _ => panic!(),
    };

    // let x = import "two" in x
    let mk_import_two = mk_import("x", "two", mk_term::var("x"), &mut vm).unwrap();
    vm.reset();
    assert_eq!(
        vm.eval(mk_import_two, &Environment::new(),)
            .map(RichTerm::without_pos)
            .unwrap(),
        mk_term::integer(2)
    );

    // let x = import "lib" in x.f
    let mk_import_lib = mk_import(
        "x",
        "lib",
        mk_term::op1(UnaryOp::StaticAccess(Ident::from("f")), mk_term::var("x")),
        &mut vm,
    );
    vm.reset();
    assert_eq!(
        vm.eval(mk_import_lib.unwrap(), &Environment::new(),)
            .map(Term::from)
            .unwrap(),
        Term::Bool(true)
    );
}

#[test]
fn interpolation_simple() {
    let mut chunks = vec![
        StrChunk::Literal(String::from("Hello")),
        StrChunk::expr(mk_term::op2(
            BinaryOp::StrConcat(),
            mk_term::string(", "),
            mk_term::string("World!"),
        )),
        StrChunk::Literal(String::from(" How")),
        StrChunk::expr(mk_term::if_then_else(
            Term::Bool(true),
            mk_term::string(" are"),
            mk_term::string(" is"),
        )),
        StrChunk::Literal(String::from(" you?")),
    ];
    chunks.reverse();

    let t: RichTerm = Term::StrChunks(chunks).into();
    assert_eq!(
        eval_no_import(t),
        Ok(Term::Str("Hello, World! How are you?".into()))
    );
}

#[test]
fn interpolation_nested() {
    let mut inner_chunks = vec![
        StrChunk::Literal(String::from(" How")),
        StrChunk::expr(
            Term::Op2(
                BinaryOp::StrConcat(),
                mk_term::string(" ar"),
                mk_term::string("e"),
            )
            .into(),
        ),
        StrChunk::expr(mk_term::if_then_else(
            Term::Bool(true),
            mk_term::string(" you"),
            mk_term::string(" me"),
        )),
    ];
    inner_chunks.reverse();

    let mut chunks = vec![
        StrChunk::Literal(String::from("Hello, World!")),
        StrChunk::expr(Term::StrChunks(inner_chunks).into()),
        StrChunk::Literal(String::from("?")),
    ];
    chunks.reverse();

    let t: RichTerm = Term::StrChunks(chunks).into();
    assert_eq!(
        eval_no_import(t),
        Ok(Term::Str("Hello, World! How are you?".into()))
    );
}

#[test]
fn initial_env() {
    let mut initial_env = Environment::new();
    let mut eval_cache = CacheImpl::new();
    initial_env.insert(
        Ident::from("g"),
        eval_cache.add(
            Closure::atomic_closure(mk_term::integer(1)),
            IdentKind::Let,
            BindingType::Normal,
        ),
    );

    let t = mk_term::let_in("x", mk_term::integer(2), mk_term::var("x"));
    assert_eq!(
        VirtualMachine::new_with_cache(DummyResolver {}, eval_cache.clone())
            .eval(t, &initial_env)
            .map(RichTerm::without_pos),
        Ok(mk_term::integer(2))
    );

    let t = mk_term::let_in("x", mk_term::integer(2), mk_term::var("g"));
    assert_eq!(
        VirtualMachine::new_with_cache(DummyResolver {}, eval_cache.clone())
            .eval(t, &initial_env)
            .map(RichTerm::without_pos),
        Ok(mk_term::integer(1))
    );

    // Shadowing of the initial environment
    let t = mk_term::let_in("g", mk_term::integer(2), mk_term::var("g"));
    assert_eq!(
        VirtualMachine::new_with_cache(DummyResolver {}, eval_cache.clone())
            .eval(t, &initial_env)
            .map(RichTerm::without_pos),
        Ok(mk_term::integer(2))
    );
}

fn mk_env(bindings: Vec<(&str, RichTerm)>, eval_cache: &mut CacheImpl) -> Environment {
    bindings
        .into_iter()
        .map(|(id, t)| {
            (
                id.into(),
                eval_cache.add(
                    Closure::atomic_closure(t),
                    IdentKind::Let,
                    BindingType::Normal,
                ),
            )
        })
        .collect()
}

#[test]
fn substitution() {
    let mut eval_cache = CacheImpl::new();
    let initial_env = mk_env(
        vec![
            ("glob1", mk_term::integer(1)),
            ("glob2", parse("\"Glob2\"").unwrap()),
            ("glob3", Term::Bool(false).into()),
        ],
        &mut eval_cache,
    );
    let env = mk_env(
        vec![
            ("loc1", Term::Bool(true).into()),
            ("loc2", parse("if glob3 then glob1 else glob2").unwrap()),
        ],
        &mut eval_cache,
    );

    let t = parse("let x = 1 in if loc1 then 1 + loc2 else glob3").unwrap();
    assert_eq!(
        subst(&eval_cache, t, &initial_env, &env),
        parse("let x = 1 in if true then 1 + (if false then 1 else \"Glob2\") else false").unwrap()
    );

    let t = parse("match{'x => [1, glob1], 'y => loc2, 'z => {id = true, other = glob3}} loc1")
        .unwrap();
    assert_eq!(
        subst(&eval_cache, t, &initial_env, &env),
        parse("match {'x => [1, 1], 'y => (if false then 1 else \"Glob2\"), 'z => {id = true, other = false}} true").unwrap()
    );
}
