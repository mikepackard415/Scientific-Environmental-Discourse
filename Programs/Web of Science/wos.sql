SELECT
    wos_id,
    issn,
    doi,
    title,
    pubtype,
    doctype,
    volume,
    issue,
    pubyear,
    pubmonth,
    pubday,
    source,
    abstract
FROM
    union_publications
WHERE
    source IN ('AMERICAN ECONOMIC REVIEW',
               'ECONOMETRICA',
               'QUARTERLY JOURNAL OF ECONOMICS',
               'REVIEW OF ECONOMIC STUDIES',
               'JOURNAL OF POLITICAL ECONOMY',
               
               'ANNUAL REVIEW OF SOCIOLOGY',
               'AMERICAN SOCIOLOGICAL REVIEW',
               'SOCIAL SCIENCE RESEARCH',
               'AMERICAN JOURNAL OF SOCIOLOGY',
               'SOCIAL SCIENCE QUARTERLY',
               
               'JOURNAL OF CONSUMER RESEARCH',
               'JOURNAL OF PEASANT STUDIES',
               'HUMAN COMMUNICATION RESEARCH',
               'SOCIAL FORCES',
               'ANTHROPOLOGICAL THEORY',
               
               'AMERICAN JOURNAL OF POLITICAL SCIENCE',
               'AMERICAN POLITICAL SCIENCE REVIEW',
               'ANNUAL REVIEW OF POLITICAL SCIENCE',
               'POLITICAL ANALYSIS',
               'BRITISH JOURNAL OF POLITICAL SCIENCE',
               
               'PSYCHOLOGICAL BULLETIN',
               'ANNUAL REVIEW OF PSYCHOLOGY',
               'PSYCHOLOGICAL SCIENCE',
               'PERSPECTIVES ON PSYCHOLOGICAL SCIENCE',
               'PSYCHOLOGICAL REVIEW');