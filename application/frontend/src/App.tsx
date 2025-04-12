// src/App.tsx
import { Routes, Route, useLocation } from "react-router-dom";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Configurations from "./pages/Configurations";
import Banner from "./components/Banner";
import Navbar from "./components/Navbar";
import Register from "./pages/Register";

export default function App() {
  const location = useLocation();
  const isLoginPage = location.pathname === "/login";
  const registerPage = location.pathname === "/register";

  return (
    <>
      <Banner />
      {(!isLoginPage && !registerPage) && <Navbar />}
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/configurations" element={<Configurations />} />
      </Routes>
    </>
  );
}
