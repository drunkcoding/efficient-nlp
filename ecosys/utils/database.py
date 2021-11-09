import mysql.connector
from mysql.connector import errorcode

from mysql.connector.connection import MySQLConnection

class DatabaseHelper(MySQLConnection):
    def __init__(self, svr_ctx=None, *args, **kwargs):
        super(DatabaseHelper, self).__init__(*args, **kwargs)
        self.svr_ctx = svr_ctx
    
    def _create_database(self, db_name):
        cursor = self.cursor()
        try:
            cursor.execute(
                "CREATE DATABASE {} DEFAULT CHARACTER SET 'utf8'".format(db_name))
        except mysql.connector.Error as err:
            self.svr_ctx.logger.fatal("Failed creating database: {}".format(err))

    def create_database(self, db_name):
        cursor = self.cursor()
        try:
            cursor.execute("USE {}".format(db_name))
        except mysql.connector.Error as err:
            self.svr_ctx.logger.info("Database {} does not exists.".format(db_name))
            if err.errno == errorcode.ER_BAD_DB_ERROR:
                self._create_database(cursor)
                self.svr_ctx.logger.info("Database {} created successfully.".format(db_name))
                self.database = db_name
            else:
                self.svr_ctx.logger.fatal(err)

    def create_table(self, table_name, table_description):
        cursor = self.cursor()
        try:
            self.svr_ctx.logger.info("Creating table {}: ".format(table_name), end='')
            cursor.execute(table_description)
        except mysql.connector.Error as err:
            if err.errno == errorcode.ER_TABLE_EXISTS_ERROR:
                self.svr_ctx.logger.warn("already exists.")
            else:
                self.svr_ctx.logger.error(err.msg)
        else:
            self.svr_ctx.logger.info("OK")
