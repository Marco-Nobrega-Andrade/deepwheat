// src/lib/api.ts
import axios from "axios";

const api = axios.create({
  baseURL: "http://localhost:8000", // ✅ Base URL for all requests
  headers: {
    "Content-Type": "application/json",
  },
});

// Optional: Add a request interceptor for auth token
api.interceptors.request.use((config) => {
  const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxIiwiZXhwIjoxNzQ0NDkwMzEzfQ.sRBvooT9YWIltxK7CQcLdq_DMB6zENnoRLNFjQzApDs"
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
