import { Request, Response, NextFunction } from "express";
import { MessageCreateParamsBase } from "@anthropic-ai/sdk/resources/messages";
import OpenAI from "openai";
import { streamOpenAIResponse } from "../utils/stream";
import { log } from "../utils/log";

/**
 * Validates OpenAI format messages to ensure complete tool_calls/tool message pairing.
 * Requires tool messages to immediately follow assistant messages with tool_calls.
 * Enforces strict immediate following sequence between tool_calls and tool messages.
 */
function validateOpenAIToolCalls(messages: any[]): any[] {
  const validatedMessages: any[] = [];
  
  for (let i = 0; i < messages.length; i++) {
    const currentMessage = { ...messages[i] };
    
    // Process assistant messages with tool_calls
    if (currentMessage.role === "assistant" && currentMessage.tool_calls) {
      const validToolCalls: any[] = [];
      const removedToolCallIds: string[] = [];
      
      // Collect all immediately following tool messages
      const immediateToolMessages: any[] = [];
      let j = i + 1;
      while (j < messages.length && messages[j].role === "tool") {
        immediateToolMessages.push(messages[j]);
        j++;
      }
      
      // For each tool_call, check if there's an immediately following tool message
      currentMessage.tool_calls.forEach((toolCall: any) => {
        const hasImmediateToolMessage = immediateToolMessages.some(toolMsg => 
          toolMsg.tool_call_id === toolCall.id
        );
        
        if (hasImmediateToolMessage) {
          validToolCalls.push(toolCall);
        } else {
          removedToolCallIds.push(toolCall.id);
          log(`Removed tool_call without immediately following tool message: ${toolCall.function?.name} (${toolCall.id})`);
        }
      });
      
      // Update the assistant message
      if (validToolCalls.length > 0) {
        currentMessage.tool_calls = validToolCalls;
      } else {
        delete currentMessage.tool_calls;
      }
      
      // Log summary if any tool_calls were removed
      if (removedToolCallIds.length > 0) {
        log(`Removed ${removedToolCallIds.length} incomplete tool_calls from assistant message`);
      }
      
      // Only include message if it has content or valid tool_calls
      if (currentMessage.content || currentMessage.tool_calls) {
        validatedMessages.push(currentMessage);
      } else {
        log(`Removed empty assistant message after tool_call cleanup`);
      }
    }
    
    // Process tool messages
    else if (currentMessage.role === "tool") {
      let hasImmediateToolCall = false;
      
      // Check if the immediately preceding assistant message has matching tool_call
      if (i > 0) {
        const prevMessage = messages[i - 1];
        if (prevMessage.role === "assistant" && prevMessage.tool_calls) {
          hasImmediateToolCall = prevMessage.tool_calls.some((toolCall: any) => 
            toolCall.id === currentMessage.tool_call_id
          );
        } else if (prevMessage.role === "tool") {
          // Check for assistant message before the sequence of tool messages
          for (let k = i - 1; k >= 0; k--) {
            if (messages[k].role === "tool") continue;
            if (messages[k].role === "assistant" && messages[k].tool_calls) {
              hasImmediateToolCall = messages[k].tool_calls.some((toolCall: any) => 
                toolCall.id === currentMessage.tool_call_id
              );
            }
            break;
          }
        }
      }
      
      if (hasImmediateToolCall) {
        validatedMessages.push(currentMessage);
      } else {
        log(`Removed tool message without immediately preceding tool_call: ${currentMessage.tool_call_id}`);
      }
    }
    
    // For all other message types, include as-is
    else {
      validatedMessages.push(currentMessage);
    }
  }
  
  return validatedMessages;
}

export const formatRequest = async (
  req: Request,
  res: Response,
  next: NextFunction
) => {
  let {
    model,
    max_tokens,
    messages,
    system = [],
    temperature,
    metadata,
    tools,
    stream,
  }: MessageCreateParamsBase = req.body;
  log("formatRequest: ", req.body);
  try {
    // @ts-ignore
    const openAIMessages = Array.isArray(messages)
      ? messages.flatMap((anthropicMessage) => {
          const openAiMessagesFromThisAnthropicMessage = [];

          if (!Array.isArray(anthropicMessage.content)) {
            // Handle simple string content
            if (typeof anthropicMessage.content === "string") {
              openAiMessagesFromThisAnthropicMessage.push({
                role: anthropicMessage.role,
                content: anthropicMessage.content,
              });
            }
            // If content is not string and not array (e.g. null/undefined), it will result in an empty array, effectively skipping this message.
            return openAiMessagesFromThisAnthropicMessage;
          }

          // Handle array content
          if (anthropicMessage.role === "assistant") {
            const assistantMessage = {
              role: "assistant",
              content: null, // Will be populated if text parts exist
            };
            let textContent = "";
            // @ts-ignore
            const toolCalls = []; // Corrected type here

            anthropicMessage.content.forEach((contentPart) => {
              if (contentPart.type === "text") {
                textContent +=
                  (typeof contentPart.text === "string"
                    ? contentPart.text
                    : JSON.stringify(contentPart.text)) + "\\n";
              } else if (contentPart.type === "tool_use") {
                toolCalls.push({
                  id: contentPart.id,
                  type: "function",
                  function: {
                    name: contentPart.name,
                    arguments: JSON.stringify(contentPart.input),
                  },
                });
              }
            });

            const trimmedTextContent = textContent.trim();
            if (trimmedTextContent.length > 0) {
              // @ts-ignore
              assistantMessage.content = trimmedTextContent;
            }
            if (toolCalls.length > 0) {
              // @ts-ignore
              assistantMessage.tool_calls = toolCalls;
            }
            // @ts-ignore
            if (
              assistantMessage.content ||
              // @ts-ignore
              (assistantMessage.tool_calls &&
                // @ts-ignore
                assistantMessage.tool_calls.length > 0)
            ) {
              openAiMessagesFromThisAnthropicMessage.push(assistantMessage);
            }
          } else if (anthropicMessage.role === "user") {
            // For user messages, text parts are combined into one message.
            // Tool results are transformed into subsequent, separate 'tool' role messages.
            let userTextMessageContent = "";
            // @ts-ignore
            const subsequentToolMessages = [];

            anthropicMessage.content.forEach((contentPart) => {
              if (contentPart.type === "text") {
                userTextMessageContent +=
                  (typeof contentPart.text === "string"
                    ? contentPart.text
                    : JSON.stringify(contentPart.text)) + "\\n";
              } else if (contentPart.type === "tool_result") {
                // Each tool_result becomes a separate 'tool' message
                subsequentToolMessages.push({
                  role: "tool",
                  tool_call_id: contentPart.tool_use_id,
                  content:
                    typeof contentPart.content === "string"
                      ? contentPart.content
                      : JSON.stringify(contentPart.content),
                });
              }
            });

            const trimmedUserText = userTextMessageContent.trim();
            if (trimmedUserText.length > 0) {
              openAiMessagesFromThisAnthropicMessage.push({
                role: "user",
                content: trimmedUserText,
              });
            }
            // @ts-ignore
            openAiMessagesFromThisAnthropicMessage.push(
              // @ts-ignore
              ...subsequentToolMessages
            );
          } else {
            // Fallback for other roles (e.g. system, or custom roles if they were to appear here with array content)
            // This will combine all text parts into a single message for that role.
            let combinedContent = "";
            anthropicMessage.content.forEach((contentPart) => {
              if (contentPart.type === "text") {
                combinedContent +=
                  (typeof contentPart.text === "string"
                    ? contentPart.text
                    : JSON.stringify(contentPart.text)) + "\\n";
              } else {
                // For non-text parts in other roles, stringify them or handle as appropriate
                combinedContent += JSON.stringify(contentPart) + "\\n";
              }
            });
            const trimmedCombinedContent = combinedContent.trim();
            if (trimmedCombinedContent.length > 0) {
              openAiMessagesFromThisAnthropicMessage.push({
                role: anthropicMessage.role, // Cast needed as role could be other than 'user'/'assistant'
                content: trimmedCombinedContent,
              });
            }
          }
          return openAiMessagesFromThisAnthropicMessage;
        })
      : [];
      
    // Validate OpenAI messages to ensure complete tool_calls/tool message pairing
    const validatedOpenAIMessages = validateOpenAIToolCalls(openAIMessages);
    
    const systemMessages =
      Array.isArray(system)
        ? system.map((item) => ({
            role: "system",
            content: [
              {
                type: "text",
                text: item.text,
                cache_control: {"type": "ephemeral"}
              }
            ]
          }))
        : [{ 
            role: "system", 
            content: [
              {
                type: "text",
                text: system,
                cache_control: {"type": "ephemeral"}
              }
            ]
          }];
    const data: any = {
      model,
      messages: [...systemMessages, ...validatedOpenAIMessages],
      temperature,
      stream,
    };
    if (tools) {
      data.tools = tools
        .filter((tool) => !["StickerRequest"].includes(tool.name))
        .map((item: any) => ({
          type: "function",
          function: {
            name: item.name,
            description: item.description,
            parameters: item.input_schema,
          },
        }));
    }
    if (stream) {
      res.setHeader("Content-Type", "text/event-stream");
    }
    res.setHeader("Cache-Control", "no-cache");
    res.setHeader("Connection", "keep-alive");
    
    // Add cache_control to the last message
    if (data.messages.length > 0) {
      const lastMessage = data.messages[data.messages.length - 1];
      if (typeof lastMessage.content === "string") {
        lastMessage.content = [
          {
            type: "text",
            text: lastMessage.content,
            cache_control: {"type": "ephemeral"}
          }
        ];
      } else if (Array.isArray(lastMessage.content)) {
        // If content is already an array, add cache_control to the last text block
        const lastContentBlock = lastMessage.content[lastMessage.content.length - 1];
        if (lastContentBlock && lastContentBlock.type === "text") {
          lastContentBlock.cache_control = {"type": "ephemeral"};
        }
      }
    }
    
    req.body = data;
    // console.log(JSON.stringify(data.messages, null, 2));
  } catch (error) {
    console.error("Error in request processing:", error);
    const errorCompletion =
      {
        async *[Symbol.asyncIterator]() {
          yield {
            id: `error_${Date.now()}`,
            created: Math.floor(Date.now() / 1000),
            model,
            object: "chat.completion.chunk",
            choices: [
              {
                index: 0,
                delta: {
                  content: `Error: ${(error as Error).message}`,
                },
                finish_reason: "stop",
              },
            ],
          };
        },
      };
    await streamOpenAIResponse(res, errorCompletion, model, req.body);
  }
  next();
};
