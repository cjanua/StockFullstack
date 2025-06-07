"use client"

import { useState, useRef, useEffect } from "react"
import { Search } from "lucide-react"

export function SearchBar() {
  const [isExpanded, setIsExpanded] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (isExpanded && inputRef.current) {
      inputRef.current.focus()
    }
  }, [isExpanded])

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsExpanded(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [])

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden rounded-full transition-all duration-300 ease-in-out ${
        isExpanded ? "w-64" : "w-10"
      } h-10 ml-auto`}
      onMouseEnter={() => setIsExpanded(true)}
      onMouseLeave={() => setIsExpanded(false)}
    >
      <div className="flex items-center h-full justify-end">
        <input
          ref={inputRef}
          type="text"
          placeholder="Search..."
          className={`absolute right-10 px-4 py-2 w-[calc(100%-2.5rem)] outline-none transition-opacity duration-300 ${
            isExpanded ? "opacity-100" : "opacity-0"
          }`}
        />
        <button
          className="absolute right-0 p-2 bg-slate-800 text-white hover:text-gray-700 focus:outline-none"
          onClick={() => {
            setIsExpanded(true)
            inputRef.current?.focus()
          }}
          aria-label="Search"
        >
          <Search className="w-5 h-5" />
        </button>
      </div>
    </div>
  )
}

