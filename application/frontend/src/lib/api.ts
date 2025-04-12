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
  const token = "token"
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export default api;
