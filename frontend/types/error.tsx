import { JSX } from "react";

export default interface Error {
    code: String;
    message: String
    fallback: JSX.Element;
}

const ERROR_CODES = ["genericError", "invalidCredentials", "invalidToken"];

const generateFallback = (code: String, message: String) => {
    const getStatement = () => {
        switch (code) {
            case "genericError":
                return "An error occurred";
            case "nextApiError":
                return "An error occurred while fetching data:";
            default:
                return "Unknown error code";
        }
    }
    return <>
        [{code}] {getStatement()}: {message}
    </>
}

export const getError = (code: String = "", message: String = ""): Error => {
    let thisCode: String = code.length > 0 ? code : "genericError";

    return {
        code: thisCode,
        message: message,
        fallback: generateFallback(thisCode, message),
    }
}
