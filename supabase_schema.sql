create extension if not exists vector with schema extensions;

create table if not exists public.documents (
    id bigint generated always as identity primary key,
    title text not null,
    source_url text not null default '',
    tax_year integer not null,
    document_type text not null default 'tax_reference',
    source_file text not null,
    source_hash text not null unique,
    embedding_model text not null,
    embedding_dimension integer not null,
    created_at timestamptz not null default now()
);

create table if not exists public.document_chunks (
    id bigint generated always as identity primary key,
    document_id bigint not null references public.documents(id) on delete cascade,
    chunk_uid text not null unique,
    chunk_index integer not null,
    content text not null,
    page_start integer not null,
    page_end integer not null,
    tax_year integer not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding extensions.vector(1024) not null,
    created_at timestamptz not null default now()
);

create index if not exists document_chunks_document_id_idx
    on public.document_chunks(document_id);

create index if not exists document_chunks_tax_year_idx
    on public.document_chunks(tax_year);

create index if not exists document_chunks_embedding_hnsw_idx
    on public.document_chunks
    using hnsw (embedding extensions.vector_cosine_ops);

create or replace function public.match_tax_chunks(
    query_embedding extensions.vector(1024),
    match_threshold double precision,
    match_count integer,
    filter_tax_year integer
)
returns table (
    id bigint,
    document_id bigint,
    title text,
    source_url text,
    content text,
    page_start integer,
    page_end integer,
    similarity double precision
)
language sql
stable
as $$
    select
        dc.id,
        dc.document_id,
        d.title,
        d.source_url,
        dc.content,
        dc.page_start,
        dc.page_end,
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.document_chunks dc
    join public.documents d on d.id = dc.document_id
    where dc.tax_year = filter_tax_year
      and 1 - (dc.embedding <=> query_embedding) >= match_threshold
    order by dc.embedding <=> query_embedding
    limit least(match_count, 50);
$$;

alter table public.documents enable row level security;
alter table public.document_chunks enable row level security;

-- The ingestion script uses the service-role key, which bypasses RLS.
-- Add narrowly scoped SELECT policies later for your chatbot's authenticated users.
