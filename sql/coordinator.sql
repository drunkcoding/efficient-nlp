CREATE DATABASE IF NOT EXISTS coordinator;

CREATE TABLE IF NOT EXISTS coordinator.model_info (
    model_name VARCHAR(255) NOT NULL,
    energy FLOAT DEFAULT NULL,
    threshold  FLOAT DEFAULT NULL,
    temperature  FLOAT DEFAULT NULL,
    labels BLOB DEFAULT NULL,
    outputs BLOB DEFAULT NULL,
    PRIMARY KEY (model_name)
);

CREATE TABLE IF NOT EXISTS coordinator.gpu_info (
    model_name VARCHAR(255) NOT NULL,
    record_time TIMESTAMP(6) DEFAULT NOW(6) ON UPDATE NOW(6),
    power FLOAT NOT NULL,
    utilization  FLOAT NOT NULL,
    mem_used  FLOAT NOT NULL,
    mem_total  FLOAT NOT NULL,
    num_query INT NOT NULL,
    batch_size INT NOT NULL,
    ctx_id INT NOT NULL
);

DELIMITER $$
DROP TRIGGER IF EXISTS energy_calculation $$
CREATE TRIGGER `energy_calculation`
AFTER INSERT ON `coordinator.gpu_info` FOR EACH ROW
BEGIN
    CREATE OR REPLACE VIEW energy AS SELECT power * (SELECT (MAX(record_time) - MIN(record_time)) / SUM(num_query) FROM coordinator.gpu_info WHERE model_name = NEW.model_name ORDER BY record_time DESC LIMIT 100) FROM coordinator.gpu_info WHERE model_name = 'distilbert' ORDER BY record_time DESC LIMIT 100;
    UPDATE coordinator.gpu_info SET energy = (SELECT AVG(Name_exp_1) FROM energy) WHERE model_name = NEW.model_name;
END $$
DELIMITER ;

-- SELECT AVG(energy) FROM (
--     SELECT power * (SELECT (MAX(record_time) - MIN(record_time)) / SUM(num_query) AS energy FROM coordinator.gpu_info WHERE model_name = 'distilbert' ORDER BY record_time DESC LIMIT 100) FROM coordinator.gpu_info WHERE model_name = 'distilbert' ORDER BY record_time DESC LIMIT 100
-- ) as t;

-- CREATE OR REPLACE VIEW energy AS
-- SELECT power * (SELECT (MAX(record_time) - MIN(record_time)) / SUM(num_query) FROM coordinator.gpu_info WHERE model_name = 'distilbert' ORDER BY record_time DESC LIMIT 100) FROM coordinator.gpu_info WHERE model_name = 'distilbert' ORDER BY record_time DESC LIMIT 100;


-- CREATE TABLE IF NOT EXISTS validation_info {
--     model_name VARCHAR(255) NOT NULL,
--     pred_labels BLOB NOT NULL,
--     true_labels BLOB NOT NULL,
--     true_labels BLOB NOT NULL,
--     FOREIGN KEY (model_name) REFERENCES model_info(model_name)
-- };
