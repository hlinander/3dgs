use duckdb::{params, Connection, Result};

pub fn init_db() -> Connection {
    let conn = Connection::open("duck.db").unwrap();
    conn.execute_batch(
        "
            CREATE SEQUENCE IF NOT EXISTS s;
            CREATE SEQUENCE IF NOT EXISTS t;
            CREATE TABLE IF NOT EXISTS grads
                (
                    id INTEGER DEFAULT nextval('s'),
                    epoch INTEGER,
                    name VARCHAR,
                    value DOUBLE
                );
            CREATE TABLE IF NOT EXISTS tensors
                (
                    id INTEGER DEFAULT nextval('t'),
                    epoch INTEGER,
                    name VARCHAR,
                    value DOUBLE
                );
        ",
    )
    .expect("");
    conn
}

pub fn log_grad(conn: &Connection, grads: Vec<f32>, name: String, epoch: usize) {
    for v in grads {
        conn.execute(
            "INSERT INTO grads (epoch, name, value) VALUES (?, ?, ?)",
            params![epoch, name, v],
        )
        .expect("");
    }
}
