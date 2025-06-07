import { JSX } from "react";

export default interface Error {
  code: string;
  message: string;
  fallback: JSX.Element;
}

const ERROR_CODES = ["genericError", "invalidCredentials", "invalidToken"];

const generateFallback = (code: string, message: string) => {
  const getStatement = () => {
    switch (code) {
      case "genericError":
        return "An error occurred";
      case "nextApiError":
        return "An error occurred while fetching data:";
      default:
        return "Unknown error code";
    }
  };
  return (
    <>
      [{code}] {getStatement()}: {message}
    </>
  );
};

export const getError = (code: string = "", message: string = ""): Error => {
  const thisCode: string = code.length > 0 ? code : ERROR_CODES[0];

  return {
    code: thisCode,
    message: message,
    fallback: generateFallback(thisCode, message),
  };
};
