import { existsSync } from "fs";
import { writeFile } from "fs/promises";
import { getOpenAICommonOptions, initConfig, initDir } from "./utils";
import { createServer } from "./server";
import { formatRequest } from "./middlewares/formatRequest";
import { rewriteBody } from "./middlewares/rewriteBody";
import { router } from "./middlewares/router";
import OpenAI from "openai";
import { streamOpenAIResponse } from "./utils/stream";
import {
  cleanupPidFile,
  isServiceRunning,
  savePid,
} from "./utils/processCheck";
import { LRUCache } from "lru-cache";
import { log } from "./utils/log";

async function initializeClaudeConfig() {
  const homeDir = process.env.HOME;
  const configPath = `${homeDir}/.claude.json`;
  if (!existsSync(configPath)) {
    const userID = Array.from(
      { length: 64 },
      () => Math.random().toString(16)[2]
    ).join("");
    const configContent = {
      numStartups: 184,
      autoUpdaterStatus: "enabled",
      userID,
      hasCompletedOnboarding: true,
      lastOnboardingVersion: "1.0.17",
      projects: {},
    };
    await writeFile(configPath, JSON.stringify(configContent, null, 2));
  }
}

interface RunOptions {
  port?: number;
}

interface ModelProvider {
  name: string;
  api_base_url: string;
  api_key: string;
  models: string[];
}

async function run(options: RunOptions = {}) {
  // Check if service is already running
  if (isServiceRunning()) {
    console.log("✅ Service is already running in the background.");
    return;
  }

  await initializeClaudeConfig();
  await initDir();
  const config = await initConfig();

  const Providers = new Map<string, ModelProvider>();
  const providerCache = new LRUCache<string, OpenAI>({
    max: 10,
    ttl: 2 * 60 * 60 * 1000,
  });

  function getProviderInstance(providerName: string): OpenAI {
    const provider: ModelProvider | undefined = Providers.get(providerName);
    if (provider === undefined) {
      throw new Error(`Provider ${providerName} not found`);
    }
    let openai = providerCache.get(provider.name);
    if (!openai) {
      openai = new OpenAI({
        baseURL: provider.api_base_url,
        apiKey: provider.api_key,
        ...getOpenAICommonOptions(),
      });
      providerCache.set(provider.name, openai);
    }
    return openai;
  }

  if (Array.isArray(config.Providers)) {
    config.Providers.forEach((provider) => {
      try {
        Providers.set(provider.name, provider);
      } catch (error) {
        console.error("Failed to parse model provider:", error);
      }
    });
  }

  if (config.OPENAI_API_KEY && config.OPENAI_BASE_URL && config.OPENAI_MODEL) {
    const defaultProvider = {
      name: "default",
      api_base_url: config.OPENAI_BASE_URL,
      api_key: config.OPENAI_API_KEY,
      models: [config.OPENAI_MODEL],
    };
    Providers.set("default", defaultProvider);
  } else if (Providers.size > 0) {
    const defaultProvider = Providers.values().next().value!;
    Providers.set("default", defaultProvider);
  }
  const port = options.port || 3456;

  // Save the PID of the background process
  savePid(process.pid);

  // Handle SIGINT (Ctrl+C) to clean up PID file
  process.on("SIGINT", () => {
    console.log("Received SIGINT, cleaning up...");
    cleanupPidFile();
    process.exit(0);
  });

  // Handle SIGTERM to clean up PID file
  process.on("SIGTERM", () => {
    cleanupPidFile();
    process.exit(0);
  });

  // Use port from environment variable if set (for background process)
  const servicePort = process.env.SERVICE_PORT
    ? parseInt(process.env.SERVICE_PORT)
    : port;

  const server = await createServer(servicePort);
  server.useMiddleware((req, res, next) => {
    req.config = config;
    next();
  });
  server.useMiddleware(rewriteBody);
  if (
    config.Router?.background &&
    config.Router?.think &&
    config?.Router?.longContext
  ) {
    server.useMiddleware(router);
  } else {
    server.useMiddleware((req, res, next) => {
      req.provider = "default";
      req.body.model = config.OPENAI_MODEL;
      next();
    });
  }
  server.useMiddleware(formatRequest);

  server.app.post("/v1/messages", async (req, res) => {
    try {
      const providerConfig: ModelProvider | undefined = Providers.get(req.provider || "default");
      if (!providerConfig) {
        throw new Error(`Provider ${req.provider || "default"} not found`);
      }

      // Prepare fetch options with proxy support
      const commonOptions = getOpenAICommonOptions();
      const fetchOptions: RequestInit = {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${providerConfig.api_key}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(req.body),
      };

      // Add proxy agent if available
      if (commonOptions.httpAgent) {
        // @ts-ignore - Node.js fetch supports agent option
        fetchOptions.agent = commonOptions.httpAgent;
      }

      const apiUrl = `${providerConfig.api_base_url}/v1/chat/completions`;
      log("Calling API:", apiUrl, "with options:", fetchOptions);
      const response = await fetch(apiUrl, fetchOptions);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle streaming vs non-streaming responses
      let completion: any;
      if (req.body.stream) {
        // For streaming, create an async iterable from the response
        completion = {
          async *[Symbol.asyncIterator]() {
            const reader = response.body?.getReader();
            const decoder = new TextDecoder();
            let buffer = '';
            
            if (!reader) {
              throw new Error("No response body reader available");
            }

            try {
              while (true) {
                const { done, value } = await reader.read();
                if (done) {
                  // Process any remaining data in buffer
                  if (buffer.trim()) {
                    const lines = buffer.split('\n');
                    for (const line of lines) {
                      if (line.trim() && line.startsWith('data: ')) {
                        const data = line.slice(6).trim();
                        if (data === '[DONE]') continue;
                        
                        try {
                          const parsed = JSON.parse(data);
                          yield parsed;
                        } catch (e) {
                          // Skip invalid JSON chunks
                          continue;
                        }
                      }
                    }
                  }
                  break;
                }
                
                // Decode chunk and add to buffer
                const chunk = decoder.decode(value, { stream: true });
                buffer += chunk;
                
                // Process complete lines from buffer
                const lines = buffer.split('\n');
                // Keep the last potentially incomplete line in buffer
                buffer = lines.pop() || '';
                
                // Process complete lines in order
                for (const line of lines) {
                  if (line.trim() && line.startsWith('data: ')) {
                    const data = line.slice(6).trim();
                    if (data === '[DONE]') continue;
                    
                    try {
                      const parsed = JSON.parse(data);
                      yield parsed;
                    } catch (e) {
                      // Skip invalid JSON chunks but log for debugging
                      log("Failed to parse SSE data:", data, e);
                      continue;
                    }
                  }
                }
              }
            } finally {
              reader.releaseLock();
            }
          }
        };
      } else {
        // For non-streaming, parse the JSON response
        completion = await response.json();
      }

      await streamOpenAIResponse(res, completion, req.body.model, req.body);
    } catch (e) {
      log("Error in direct API call:", e);
    }
  });
  server.start();
  console.log(`🚀 Claude Code Router is running on port ${servicePort}`);
}

export { run };
// run();
