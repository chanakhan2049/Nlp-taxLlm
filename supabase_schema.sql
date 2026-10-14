create extension if not exists vector with schema extensions;

create table if not exists public.documents (
    id bigint generated always as identity primary key,
    title text not null,
    tax_year integer not null,
    document_year integer not null,
    document_type text not null default 'tax_reference',
    source_file text not null,
    source_hash text not null unique,
    embedding_model text not null,
    embedding_dimension integer not null,
    created_at timestamptz not null default now()
);

-- Migration for databases created before document year and applicability year
-- were stored separately. Existing rows retain their former year until the
-- source documents are re-ingested.
alter table public.documents
    add column if not exists document_year integer;
update public.documents
set document_year = tax_year
where document_year is null;
alter table public.documents
    alter column document_year set not null;

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

-- These source documents remain applicable for the 2568 dataset even when
-- they were published earlier. Preserve the original year separately, then
-- make all existing records discoverable by the current-year filter.
update public.documents
set title = replace(
        title,
        'ปีภาษี ' || document_year,
        'เอกสารเผยแพร่ปี ' || document_year
    ),
    tax_year = 2568;

update public.document_chunks as dc
set tax_year = 2568,
    metadata = dc.metadata || jsonb_build_object(
        'title', d.title,
        'document_year', d.document_year,
        'applicable_tax_year', 2568
    )
from public.documents as d
where d.id = dc.document_id;

create index if not exists document_chunks_document_id_idx
    on public.document_chunks(document_id);

create index if not exists document_chunks_tax_year_idx
    on public.document_chunks(tax_year);

create index if not exists document_chunks_embedding_hnsw_idx
    on public.document_chunks
    using hnsw (embedding extensions.vector_cosine_ops);

-- PostgreSQL cannot change an existing function's returned columns with
-- CREATE OR REPLACE, so remove the old four-argument version first.
drop function if exists public.match_tax_chunks(
    extensions.vector,
    double precision,
    integer,
    integer
);

create function public.match_tax_chunks(
    query_embedding extensions.vector(1024),
    match_threshold double precision,
    match_count integer,
    filter_tax_year integer
)
returns table (
    id bigint,
    document_id bigint,
    title text,
    document_year integer,
    tax_year integer,
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
        d.document_year,
        dc.tax_year,
        dc.content,
        dc.page_start,
        dc.page_end,
        1 - (dc.embedding <=> query_embedding) as similarity
    from public.document_chunks dc
    join public.documents d on d.id = dc.document_id
    where dc.tax_year = filter_tax_year
      and coalesce(dc.metadata->>'text_quality', 'ok') <> 'corrupt'
      and 1 - (dc.embedding <=> query_embedding) >= match_threshold
    order by dc.embedding <=> query_embedding
    limit least(match_count, 50);
$$;

alter table public.documents enable row level security;
alter table public.document_chunks enable row level security;

-- The ingestion script uses the service-role key, which bypasses RLS.
-- Add narrowly scoped SELECT policies later for your chatbot's authenticated users.
