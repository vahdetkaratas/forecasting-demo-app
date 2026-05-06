# Forecasting demo — multi-profile shell

Static HTML shell sources for **forecasting.vahdetkaratas.com** (recruiter / CV / proof-of-work) and **forecasting.vahdetlabs.com** (commercial context). The runnable app is **Streamlit only**: `https://forecast.vahdetkaratas.com/`.

## Source layout

| Path | Role |
|------|------|
| `index.html` | Template with `{{PLACEHOLDER}}` tokens |
| `shell.css` / `demo-content.css` / `shell.js` | Shared chrome + body typography |
| `render-shell.mjs` | Node renderer: merges profile + project JSON + body HTML |
| `profile.json` | Default fallback profile if `profiles/<name>.json` is missing |
| `profiles/recruiter.json` | Identity + sidebar copy for personal / hiring domain |
| `profiles/commercial.json` | Identity + sidebar copy for Vahdetlabs domain |
| `projects/forecasting.json` | Hero, CTAs, limitations, profile-specific overrides |
| `body/forecasting.html` | Technical narrative sections |

## Render commands

From the **repository root** (parent of `shell/`):

```bash
node shell/render-shell.mjs --project shell/projects/forecasting.json --body shell/body/forecasting.html --out layout-shell --profile recruiter

node shell/render-shell.mjs --project shell/projects/forecasting.json --body shell/body/forecasting.html --out layout-shell-commercial --profile commercial
```

Outputs:

- `layout-shell/` — deploy at **forecasting.vahdetkaratas.com**
- `layout-shell-commercial/` — deploy at **forecasting.vahdetlabs.com**

Each folder contains `index.html`, `shell.css`, `demo-content.css`, `shell.js`, `favicon.svg`, `profile.json`.

## Rules

- Do not edit `layout-shell*` by hand as source — change `shell/` and re-run Node.
- Do not claim REST APIs, RAG, or KPI dashboards here; this repo is a Streamlit forecasting MVP.
