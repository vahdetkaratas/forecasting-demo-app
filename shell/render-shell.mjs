import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_PROFILE = "recruiter";

const GH_ICON = `<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/></svg>`;

function usage() {
  console.error(
    "Usage: node shell/render-shell.mjs --project <project.json> --body <body.html> --out <output-dir> [--profile <profile-name>]"
  );
  process.exit(1);
}

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i += 1) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!key.startsWith("--") || !value) usage();
    args[key.slice(2)] = value;
    i += 1;
  }
  return args;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function listItems(items) {
  return items.map((item) => `<li>${escapeHtml(item)}</li>`).join("\n");
}

function linkItems(items) {
  return items
    .map(
      (item) =>
        `<li><a href="${escapeHtml(item.href)}"${item.external ? ' target="_blank" rel="noopener noreferrer"' : ""}>${escapeHtml(item.label)}</a></li>`
    )
    .join("\n");
}

function socialItems(items) {
  return items
    .map(
      (item) =>
        `<a href="${escapeHtml(item.href)}" target="_blank" rel="noopener noreferrer" title="${escapeHtml(item.label)}" aria-label="${escapeHtml(item.label)}"><i class="${escapeHtml(item.iconClass)}" aria-hidden="true"></i></a>`
    )
    .join("\n");
}

function ctaButtons(items) {
  if (!items?.length) return "";
  return items
    .map((item, index) => {
      const primary = index === 0;
      const cls = primary ? "btn btn-primary" : "btn btn-ghost";
      const ext = item.external ? ' target="_blank" rel="noopener noreferrer"' : "";
      const gh =
        primary &&
        typeof item.href === "string" &&
        item.href.includes("github.com");
      const label = escapeHtml(item.label);
      const inner = gh ? `${GH_ICON}${label}` : label;
      return `<a class="${cls}" href="${escapeHtml(item.href)}"${ext}>${inner}</a>`;
    })
    .join("\n");
}

function metricsFromStack(stack) {
  if (!stack?.length) return "";
  return stack
    .slice(0, 4)
    .map(
      (s) =>
        `<div><span>${escapeHtml(s)}</span></div>`
    )
    .join("\n");
}

function limitationsSection(items) {
  if (!items?.length) return "";
  const lis = items.map((t) => `<li>${escapeHtml(t)}</li>`).join("\n");
  return `<section id="scope" aria-labelledby="h-scope"><h2 id="h-scope">Limitations &amp; scope</h2><div class="pitch"><ul>${lis}</ul></div></section>`;
}

function heroNoteHtml(note) {
  const n = note && String(note).trim();
  if (!n) return "";
  return `<p class="hero-note">${escapeHtml(n)}</p>`;
}

function identitySubHtml(profile) {
  const sub = profile.identitySub && String(profile.identitySub).trim();
  if (!sub) return "";
  return `<div class="art-sm-text art-muted mb-10">${escapeHtml(sub)}</div>`;
}

function relatedGroupsHtml(project) {
  const ml = project.relatedMlSystems || [];
  const dt = project.relatedDataTools || [];
  const legacy = project.relatedSystems || [];
  if (!ml.length && !dt.length && legacy.length) {
    return `<div class="art-knowledge-block p-30-15"><h6 class="mb-15">Related</h6><ul class="art-knowledge-list p-15-0">${linkItems(legacy)}</ul></div>`;
  }
  let html = "";
  if (ml.length) {
    html += `<div class="art-knowledge-block p-30-15"><h6 class="mb-15">ML systems</h6><ul class="art-knowledge-list p-15-0">${linkItems(ml)}</ul></div>`;
  }
  if (dt.length) {
    html += `<div class="art-knowledge-block p-30-15"><h6 class="mb-15">Data tools</h6><ul class="art-knowledge-list p-15-0">${linkItems(dt)}</ul></div>`;
  }
  return html;
}

function mergeProject(baseProject, profileName) {
  const variant = baseProject.profiles?.[profileName] || {};
  return {
    ...baseProject,
    ...variant,
  };
}

async function readJson(filePath) {
  return JSON.parse(await fs.readFile(filePath, "utf8"));
}

async function readProfile(profileName) {
  const profilePath = path.join(ROOT, "profiles", `${profileName}.json`);
  try {
    return await readJson(profilePath);
  } catch (error) {
    if (profileName !== DEFAULT_PROFILE) {
      throw new Error(
        `Profile '${profileName}' was not found at ${profilePath}.`
      );
    }

    return readJson(path.join(ROOT, "profile.json"));
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const projectPath = args.project ? path.resolve(args.project) : null;
  const bodyPath = args.body ? path.resolve(args.body) : null;
  const outDir = args.out ? path.resolve(args.out) : null;
  const profileName = args.profile || DEFAULT_PROFILE;

  if (!projectPath || !bodyPath || !outDir) usage();

  const profile = await readProfile(profileName);
  const projectBase = await readJson(projectPath);
  const project = mergeProject(projectBase, profileName);
  const template = await fs.readFile(path.join(ROOT, "index.html"), "utf8");
  const bodyHtml = await fs.readFile(bodyPath, "utf8");

  const pageTitle =
    project.pageTitle || `${project.title} - ${profile.name}`;
  const metaDescription = project.metaDescription || project.summary;
  const themeColor = project.themeColor || "#1a1a1f";
  const logo1 = project.logoLine1 || "Forecast";
  const logo2 = project.logoLine2 || "Demo shell";
  const heroTitle = project.heroTitle || project.title;
  const footerTitle = project.footerProjectName || "Forecasting demo";
  const footerSub =
    project.footerProjectSub ||
    "Streamlit · pandas · statsmodels";

  const rendered = template
    .replaceAll("{{PAGE_TITLE}}", escapeHtml(pageTitle))
    .replaceAll("{{META_DESCRIPTION}}", escapeHtml(metaDescription))
    .replaceAll("{{THEME_COLOR}}", escapeHtml(themeColor))
    .replaceAll("{{PORTFOLIO_URL}}", escapeHtml(profile.portfolioUrl))
    .replaceAll("{{AVATAR_URL}}", escapeHtml(profile.avatarUrl))
    .replaceAll("{{PROFILE_NAME}}", escapeHtml(profile.name))
    .replaceAll("{{PROFILE_IDENTITY}}", escapeHtml(profile.identity))
    .replaceAll("{{PROFILE_IDENTITY_SUB_HTML}}", identitySubHtml(profile))
    .replaceAll("{{PROFILE_LOCATION}}", escapeHtml(profile.location))
    .replaceAll(
      "{{TECHNICAL_FOCUS_ITEMS}}",
      listItems(profile.technicalFocus || [])
    )
    .replaceAll(
      "{{REVIEW_SECTION_TITLE}}",
      escapeHtml(profile.reviewSectionTitle || "Review")
    )
    .replaceAll(
      "{{REVIEW_LINK_ITEMS}}",
      linkItems(project.reviewLinks || [])
    )
    .replaceAll("{{RELATED_GROUPS_HTML}}", relatedGroupsHtml(project))
    .replaceAll(
      "{{SOCIAL_LINK_ITEMS}}",
      socialItems(profile.socialLinks || [])
    )
    .replaceAll("{{LOGO_LINE_1}}", escapeHtml(logo1))
    .replaceAll("{{LOGO_LINE_2}}", escapeHtml(logo2))
    .replaceAll("{{NAV_BUTTONS}}", ctaButtons(project.projectLinks || []))
    .replaceAll(
      "{{PROJECT_EYEBROW}}",
      escapeHtml(project.eyebrow || "Portfolio artifact")
    )
    .replaceAll("{{PROJECT_HERO_TITLE}}", escapeHtml(heroTitle))
    .replaceAll("{{PROJECT_SUMMARY}}", escapeHtml(project.summary))
    .replaceAll(
      "{{HERO_CTA_BUTTONS}}",
      ctaButtons(project.projectLinks || [])
    )
    .replaceAll("{{HERO_NOTE_HTML}}", heroNoteHtml(project.heroNote))
    .replaceAll("{{METRICS_ITEMS}}", metricsFromStack(project.stack || []))
    .replaceAll(
      "{{LIMITATIONS_SECTION_HTML}}",
      limitationsSection(project.limitations || [])
    )
    .replaceAll("{{CONTENT_HTML}}", bodyHtml)
    .replaceAll("{{PROJECT_FOOTER_TITLE}}", escapeHtml(footerTitle))
    .replaceAll("{{PROJECT_FOOTER_SUB}}", escapeHtml(footerSub))
    .replaceAll(
      "{{FOOTER_TAGLINE}}",
      escapeHtml(
        profile.footerTagline ||
          "Time-series tooling and interactive dashboards."
      )
    );

  await fs.rm(outDir, { recursive: true, force: true });
  await fs.mkdir(outDir, { recursive: true });
  await fs.writeFile(path.join(outDir, "index.html"), rendered, "utf8");

  for (const file of ["shell.css", "demo-content.css", "shell.js", "favicon.svg"]) {
    const src = path.join(ROOT, file);
    try {
      await fs.copyFile(src, path.join(outDir, file));
    } catch (e) {
      if (file !== "favicon.svg") throw e;
    }
  }

  await fs.writeFile(
    path.join(outDir, "profile.json"),
    `${JSON.stringify(profile, null, 2)}\n`,
    "utf8"
  );

  console.log(`Rendered shell to ${outDir} using profile '${profileName}'`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
