import React from 'react';


const Rocket = () => (
    <svg width="100" height="200" viewBox="0 0 100 200" xmlns="http://www.w3.org/2000/svg">
        <circle cx="50" cy="50" r="40" fill="red" />
        <circle cx="50" cy="150" r="40" fill="red" />
        <ellipse cx="50" cy="100" rx="30" ry="60" fill="blue" />
    </svg>
);

export default async function Home() {

    return <>
      {Rocket()}
    </>;
  }