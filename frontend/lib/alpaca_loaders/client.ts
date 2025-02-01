import { createClient, CreateClientOptions, Client } from "@alpacahq/typescript-sdk";
import { env } from "process";

// const ALPACA_KEY = env.APCA_KEY_ID;
// const ALPACA_SECRET = env.APCA_KEY_SECRET;
const ALPCA_URL = env.APCA_URL;

export default class AlpacaClient {
  private static instance: AlpacaClient = new AlpacaClient();
  private client: Client;

  private constructor() {
    // console.log(ALPACA_KEY, ALPACA_SECRET);
    const opts: CreateClientOptions = {
      paper: false,
      baseURL: ALPCA_URL,
    };
    this.client = createClient(opts);
  }

  public static getClient(): Client {
    return AlpacaClient.instance.client;
  }
}