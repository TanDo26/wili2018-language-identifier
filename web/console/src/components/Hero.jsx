const TAGS = ['Text Classification', 'language-identification', 'License: ODbL', 'arXiv 1801.07779'];

export default function Hero() {
  return (
    <section aria-labelledby="hero-title">
      <h1
        id="hero-title"
        className="max-w-2xl text-[32px] font-bold leading-[1.15] text-ink sm:text-[40px]"
      >
        Wikipedia Language Identification Benchmark
      </h1>
      <p className="mt-3 max-w-2xl text-base text-muted">
        Paragraphs from Wikipedia, labeled by language. Browse samples, inspect data splits, and
        evaluate model pipelines.
      </p>
      <ul aria-label="Dataset tags" className="mt-4 flex flex-wrap gap-2">
        {TAGS.map((tag) => (
          <li key={tag} className="rounded-full bg-chip px-3 py-1 font-mono text-xs text-[#2B4442]">
            {tag}
          </li>
        ))}
      </ul>
    </section>
  );
}
