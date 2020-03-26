#![feature(proc_macro_hygiene, decl_macro)]

use std::path::PathBuf;
use std::path::Path;
use std::fs::File;
use std::io::prelude::*;

use rocket::response::content::Html;
use rocket::response::NamedFile;
#[macro_use] extern crate rocket;

mod expectation_maximization;

#[get("/<_path..>")]
fn index_path(_path: PathBuf) -> Html<String> {
    let mut file = File::open("../client/dist/index.html").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();
    Html(contents)
}

#[get("/static/<file..>")]
fn files(file: PathBuf) -> Option<NamedFile> {
    NamedFile::open(Path::new("../client/dist/static/").join(file)).ok()
}

#[get("/")]
fn index() -> Html<String> {
    return index_path(PathBuf::new())
}