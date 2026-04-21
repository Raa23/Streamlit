CREATE TABLE news_articles (
    id              BIGSERIAL PRIMARY KEY,
    url             TEXT NOT NULL UNIQUE,          -- natural dedup key
    title           TEXT NOT NULL,
    source          TEXT,
    source_domain   TEXT,
    category        TEXT,
    summary         TEXT,
    banner_image    TEXT,
    authors         TEXT[],                        -- simple array
    time_published  TIMESTAMPTZ NOT NULL,
    sentiment_score REAL,
    sentiment_label TEXT,
    topics          JSONB,                         -- [{topic, relevance_score}] — rarely queried by field
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Normalized: you'll want to query/filter by ticker
CREATE TABLE news_ticker_sentiment (
    id              BIGSERIAL PRIMARY KEY,
    article_id      BIGINT NOT NULL REFERENCES news_articles(id) ON DELETE CASCADE,
    ticker          TEXT NOT NULL,
    relevance_score REAL,
    sentiment_score REAL,
    sentiment_label TEXT
);

CREATE INDEX idx_ticker_sentiment_ticker ON news_ticker_sentiment(ticker);
CREATE INDEX idx_ticker_sentiment_article ON news_ticker_sentiment(article_id);
CREATE INDEX idx_articles_time ON news_articles(time_published DESC);
CREATE INDEX idx_articles_sentiment ON news_articles(sentiment_label);