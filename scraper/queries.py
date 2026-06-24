"""
GraphQL queries for scraping merged PRs, reviews, threads, and comments.

All queries use cursor-based pagination.  The scraper calls these with
variables filled in at runtime.
"""

# ── Merged PRs (paginated) ───────────────────────────────────────────

MERGED_PRS_QUERY = """
query MergedPRs($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      states: MERGED
      first: 50
      after: $cursor
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        id
        number
        title
        author { login }
        state
        createdAt
        mergedAt
        closedAt
        baseRefName
        headRefName
        mergeCommit { oid }
        body
        labels(first: 20) { nodes { name } }
        additions
        deletions
        changedFiles
        reviewDecision
        url
      }
    }
  }
}
"""

# ── Closed PRs (paginated) ──────────────────────────────────────────

CLOSED_PRS_QUERY = """
query ClosedPRs($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequests(
      states: CLOSED
      first: 50
      after: $cursor
      orderBy: {field: UPDATED_AT, direction: DESC}
    ) {
      totalCount
      pageInfo { hasNextPage endCursor }
      nodes {
        id
        number
        title
        author { login }
        state
        createdAt
        mergedAt
        closedAt
        baseRefName
        headRefName
        mergeCommit { oid }
        body
        labels(first: 20) { nodes { name } }
        additions
        deletions
        changedFiles
        reviewDecision
        url
      }
    }
  }
}
"""

# ── Commits for a single PR ─────────────────────────────────────────

PR_COMMITS_QUERY = """
query PRCommits($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      commits(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          commit {
            oid
            message
            authoredDate
            author { user { login } }
            parents(first: 10) { nodes { oid } }
          }
        }
      }
    }
  }
}
"""

# ── Files changed in a PR ───────────────────────────────────────────

PR_FILES_QUERY = """
query PRFiles($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      files(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          path
          additions
          deletions
          changeType
        }
      }
    }
  }
}
"""

# ── Reviews for a PR ────────────────────────────────────────────────

PR_REVIEWS_QUERY = """
query PRReviews($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviews(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          author { login }
          submittedAt
          state
          body
          commit { oid }
        }
      }
    }
  }
}
"""

# ── Review threads + inline comments (the key query) ────────────────
# This is why we need GraphQL: thread resolution state is not in REST.

PR_REVIEW_THREADS_QUERY = """
query PRReviewThreads($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 50, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          isResolved
          isOutdated
          resolvedBy { login }
          path
          line
          originalLine
          startLine
          originalStartLine
          diffSide
          subjectType
          comments(first: 100) {
            nodes {
              id
              author { login }
              body
              createdAt
              updatedAt
              replyTo { id }
              path
              line
              originalLine
              commit { oid }
              originalCommit { oid }
              diffHunk
              pullRequestReview {
                id
                state
              }
            }
          }
        }
      }
    }
  }
}
"""

# ── Issue-level (top-level) comments on a PR ────────────────────────

PR_ISSUE_COMMENTS_QUERY = """
query PRIssueComments($owner: String!, $name: String!, $number: Int!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      comments(first: 100, after: $cursor) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          author { login }
          body
          createdAt
          updatedAt
        }
      }
    }
  }
}
"""
