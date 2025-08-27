import mysql from "mysql2/promise";
import dotenv from 'dotenv';
dotenv.config({path:'../.env'})

export const pool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASS,
  database: process.env.DB_NAME,
  waitForConnections: true,
  connectionLimit: 10,
});
console.log(process.env.DB_PASS)