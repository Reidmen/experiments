use actix_web::{dev::Server, Error};
use std::{fmt::format, net::TcpListener};
use subscriber::execute;

#[tokio::main]
async fn main() -> () {
    let listener = TcpListener::bind("127.0.0.1:0").expect("Failed at binding at random port");
    let port = listener.local_addr().unwrap().port();
    let server = execute(listener).expect("Failed to bind to address");
    format!("port {}", port);
}
