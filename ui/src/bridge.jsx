import React from "react";
import { createRoot } from "react-dom/client";
import AssistantMarkdown from "./AssistantMarkdown.jsx";

const roots = new WeakMap();

function mount(el, props) {
  if (!el) return;
  let root = roots.get(el);
  if (!root) {
    root = createRoot(el);
    roots.set(el, root);
  }
  root.render(<AssistantMarkdown {...props} />);
}

function unmount(el) {
  const root = roots.get(el);
  if (root) {
    root.unmount();
    roots.delete(el);
  }
}

export default { mount, unmount };
