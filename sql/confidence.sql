CREATE DATABASE IF NOT EXISTS coordinator;

CREATE TABLE IF NOT EXISTS model_info {
    model_id VARCHAR(128) NOT NULL,
    model_name VARCHAR(255) NOT NULL,
    model_size INT NOT NULL,
    threashold  FLOAT DEFAULT -1 NOT NULL,
    PRIMARY KEY (model_id)
};