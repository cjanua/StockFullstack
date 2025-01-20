// "use client";

// import { useState } from "react";
// import { useRouter } from "next/navigation";

// export default function RegistrationForm({ userId }: { userId: string }) {
//   const router = useRouter();
//   const [email, setEmail] = useState("");
//   const [password, setPassword] = useState("");
//   const [error, setError] = useState("");

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();

//     try {
//       const res = await fetch("/api/auth/complete-registration", {
//         method: "POST",
//         headers: { "Content-Type": "application/json" },
//         body: JSON.stringify({ userId, email, password }),
//       });

//       if (!res.ok) throw new Error("Registration failed");

//       router.push("/dashboard");
//     } catch (err) {
//       setError("Failed to complete registration");
//     }
//   };

//   return (
//     <form onSubmit={handleSubmit} className="space-y-4">
//       <div>
//         <label className="block mb-1">Email</label>
//         <input
//           type="email"
//           value={email}
//           onChange={(e) => setEmail(e.target.value)}
//           className="w-full p-2 border rounded"
//           required
//         />
//       </div>
//       <div>
//         <label className="block mb-1">Password</label>
//         <input
//           type="password"
//           value={password}
//           onChange={(e) => setPassword(e.target.value)}
//           className="w-full p-2 border rounded"
//           required
//         />
//       </div>
//       {error && <p className="text-red-500">{error}</p>}
//       <button
//         type="submit"
//         className="w-full bg-blue-500 text-white p-2 rounded hover:bg-blue-600"
//       >
//         Complete Registration
//       </button>
//     </form>
//   );
// }
