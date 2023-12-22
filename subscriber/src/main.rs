use subscriber::execute;

#[tokio::main]
async fn main() -> std::io::Result<()> {
    execute("17.0.0.1:0")?.await
}
