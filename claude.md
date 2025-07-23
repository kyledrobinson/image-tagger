# 🧠 CLAUDE.md

*A Codebase Guide That Doesn’t Suck™*

> Designed for speed, clarity, and your future self.

---

## 🚀 Project Setup & Dev Flow

### Local Dev Quickstart

```bash
# Clone it like you mean it
git clone https://github.com/yourname/yourproject.git && cd yourproject

# Virtualenv setup (if Python)
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# Node project setup (if JS)
npm install
```

### Docker (Optional/For Deployment)

```bash
# Build it
docker build . -t yourproject:latest

# Run it
docker run -it -p 8080:8080 yourproject:latest
```

---

## 🧪 Quality Control

### Testing & Linting

* ✅ **Lint**: `ruff check .` / `eslint .` (depending on language)
* 🎨 **Format**: `black .` or `prettier --write .`
* 🔍 **Typecheck**: `pyright` or `tsc --noEmit`
* 🧚 **Tests**:

  * All: `pytest -v` / `npm test`
  * Single test: `pytest tests/something.py::test_specific`
* 🧚 **Visual Check** (for UI):

  * `npm run storybook`
  * Or open `index.html` and eyeball like a boss 👀

---

## 🧱 Code Structure & Style

### Python

* 🐍 `snake_case` for vars/functions
* 🏧 `PascalCase` for classes
* 🧼 Dataclasses preferred when modeling clean data
* ❌ No try/except blocks without logging or raising
* ✔️ Custom errors > cryptic ones (`raise ToolError("Bad monkey")`)

### JS / TS

* 🧠 `Strict mode` enabled
* 🧱 Function components with Hooks
* 🛠 Clear state: `useState`, `useReducer`, or Zustand
* 📀 Typed interfaces > loose `any`
* 🌈 Styled with `Tailwind` or `CSS Modules`, unless you’re feeling sassy

---

## 🛍 Architecture Practices

* ✅ **Plan-first**: Don’t code blind. Plan your route.
* 💣 **No dead code**: If it’s not used, it's gone.
* ❌ No versioned functions: `processV2` is your enemy.
* 🔎 Explicit > clever: If you have to explain it in Slack, refactor it.

### File Structure Rules

```
/src
  /components    → Reusable UI
  /hooks         → Custom logic
  /utils         → Pure functions only
  /pages         → Route-level code
  /styles        → Tailwind or global CSS
```

---

## 💥 Feature Flow: The Kyle Method™

```text
1. Research  → 2. Plan  → 3. Implement  → 4. Validate
```

**Start with:**

> "Let me scan the code and map out a plan before writing anything."

* Plan feature and file locations
* Sketch interface or API changes
* Break work into micro-commits
* Write once, cry never

---

## ⚔️ Debugging Rules of Engagement

* If it’s flaky, isolate it.
* If it’s unclear, rename it.
* If it smells bad, split it up.
* If you’re stuck, stop & say:

  > "Let me ultrathink this architecture."

---

## 📊 Testing Rules

* Complex logic → write tests first
* CRUD → write tests after, keep 'em lean
* UI → test core flows, not pixels
* Hot paths → benchmark before "optimizing"

---

## 🛡 Security Checklist (Mini Edition™)

* Input validation? ✅
* No exposed secrets? ✅
* Rate-limiting (if public API)? ✅
* Auth? Don't reinvent. Use a lib.

---

## 📋 Dev Commands Cheat Sheet

| Task              | Command                          |
| ----------------- | -------------------------------- |
| Setup (Python)    | `./setup.sh` or manual           |
| Format Python     | `black . && ruff format .`       |
| Typecheck TS      | `tsc --noEmit`                   |
| Run Dev (Next.js) | `npm run dev`                    |
| Build             | `npm run build` / `docker build` |
| Test              | `pytest` / `npm test`            |

---

## 🧠 Philosophy

* **Code like someone else will read it...**
  …and that someone is tired, undercaffeinated, and future-you.

* **Delete fearlessly.**
  If it’s not in use and not explained, it’s gone.

* **Automate what’s boring.**
  Repeat tasks = write a script.

* **Design is logic you can see.**
  Make it flow beautifully—frontend or backend.

---
