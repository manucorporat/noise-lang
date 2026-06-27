import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// The blog lives in src/content/blog/*.{md,mdx}. Each file's name (minus extension) is its slug,
// so `src/content/blog/hello-noise.mdx` → /blog/hello-noise. Posts are plain Markdown for prose
// or MDX when they want to embed live components (CodePanel, the scrollytelling demos, …).
const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    // A `YYYY-MM-DD` in frontmatter is coerced to a Date here, so posts can sort chronologically.
    date: z.coerce.date(),
    description: z.string(),
    author: z.string().default('Manu MA'),
    tags: z.array(z.string()).default([]),
    // Drafts build locally but are filtered out of the index/listing (see pages/blog).
    draft: z.boolean().default(false),
  }),
});

export const collections = { blog };
