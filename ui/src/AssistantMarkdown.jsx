import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";

const markdownComponents = {
  table: ({ children, ...props }) => (
    <div className="chat-md-table-wrap">
      <table {...props}>{children}</table>
    </div>
  ),
  a: ({ children, ...props }) => (
    <a {...props} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  ),
};

export default function AssistantMarkdown({ markdown, streaming }) {
  return (
    <div className="chat-md">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={markdownComponents}
      >
        {markdown || ""}
      </ReactMarkdown>
      {streaming ? (
        <span className="chat-md-cursor" aria-hidden="true" />
      ) : null}
    </div>
  );
}
