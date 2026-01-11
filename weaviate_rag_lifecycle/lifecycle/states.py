from enum import Enum

class LifecycleState(str, Enum):
    DRAFT = 'draft'
    INDEXING = 'indexing'
    STAGING = 'staging'
    PRODUCTION = 'production'
    DEPRECATED = 'deprecated'
    ARCHIVED = 'archived'
