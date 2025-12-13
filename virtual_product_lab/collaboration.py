"""
Collaboration Hub Module
========================

Real-time collaboration tools for multi-discipline product development.
Enables value network coordination across teams, suppliers, and stakeholders.

Copyright (c) Joshua Hendricks Cole (DBA: Corporation of Light)
PATENT PENDING - All Rights Reserved
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from enum import Enum, auto
import uuid
import json


class StakeholderRole(Enum):
    """Roles in the product development value network."""
    PROGRAM_MANAGER = auto()
    DESIGN_ENGINEER = auto()
    SYSTEMS_ENGINEER = auto()
    MECHANICAL_ENGINEER = auto()
    ELECTRICAL_ENGINEER = auto()
    SOFTWARE_ENGINEER = auto()
    MANUFACTURING_ENGINEER = auto()
    QUALITY_ENGINEER = auto()
    PROCUREMENT_SPECIALIST = auto()
    SUPPLIER = auto()
    CUSTOMER = auto()
    REGULATORY_SPECIALIST = auto()
    TEST_ENGINEER = auto()
    EXECUTIVE_SPONSOR = auto()


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()


class TaskStatus(Enum):
    """Task status states."""
    NOT_STARTED = auto()
    IN_PROGRESS = auto()
    BLOCKED = auto()
    IN_REVIEW = auto()
    COMPLETED = auto()
    CANCELLED = auto()


class ChangeType(Enum):
    """Types of design changes."""
    ENGINEERING_CHANGE = auto()
    DEVIATION = auto()
    WAIVER = auto()
    SUPPLIER_CHANGE = auto()
    SPECIFICATION_CHANGE = auto()
    PROCESS_CHANGE = auto()


@dataclass
class Stakeholder:
    """A stakeholder in the product development process."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    email: str = ""
    organization: str = ""
    department: str = ""
    role: StakeholderRole = StakeholderRole.DESIGN_ENGINEER
    disciplines: List[str] = field(default_factory=list)

    # Access control
    can_edit: bool = False
    can_approve: bool = False
    can_release: bool = False

    # Activity
    last_active: datetime = field(default_factory=datetime.now)
    assigned_components: List[str] = field(default_factory=list)
    assigned_tasks: List[str] = field(default_factory=list)

    # Communication preferences
    notification_email: bool = True
    notification_app: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'organization': self.organization,
            'department': self.department,
            'role': self.role.name,
            'disciplines': self.disciplines,
            'can_edit': self.can_edit,
            'can_approve': self.can_approve,
            'can_release': self.can_release,
            'last_active': self.last_active.isoformat(),
            'assigned_components': self.assigned_components,
            'assigned_tasks': self.assigned_tasks
        }


@dataclass
class CollaborationTask:
    """A task in the collaboration workflow."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    description: str = ""
    task_type: str = "design"  # design, review, test, manufacture, etc.

    # Assignment
    assigned_to: List[str] = field(default_factory=list)  # stakeholder IDs
    created_by: str = ""

    # Related items
    related_components: List[str] = field(default_factory=list)
    related_documents: List[str] = field(default_factory=list)

    # Timeline
    created_date: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None

    # Status
    priority: TaskPriority = TaskPriority.MEDIUM
    status: TaskStatus = TaskStatus.NOT_STARTED
    percent_complete: int = 0

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # task IDs
    blocks: List[str] = field(default_factory=list)  # task IDs

    # Comments
    comments: List[Dict[str, Any]] = field(default_factory=list)

    def add_comment(self, author_id: str, text: str):
        """Add a comment to the task."""
        self.comments.append({
            'id': str(uuid.uuid4())[:8],
            'author_id': author_id,
            'text': text,
            'timestamp': datetime.now().isoformat()
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'task_type': self.task_type,
            'assigned_to': self.assigned_to,
            'created_by': self.created_by,
            'related_components': self.related_components,
            'related_documents': self.related_documents,
            'created_date': self.created_date.isoformat(),
            'due_date': self.due_date.isoformat() if self.due_date else None,
            'completed_date': self.completed_date.isoformat() if self.completed_date else None,
            'priority': self.priority.name,
            'status': self.status.name,
            'percent_complete': self.percent_complete,
            'depends_on': self.depends_on,
            'blocks': self.blocks,
            'comments': self.comments
        }


@dataclass
class DesignReview:
    """A design review event."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    review_type: str = "technical"  # technical, peer, gate, customer
    description: str = ""

    # Participants
    reviewers: List[str] = field(default_factory=list)  # stakeholder IDs
    presenter: str = ""

    # Scope
    components_in_scope: List[str] = field(default_factory=list)
    documents: List[str] = field(default_factory=list)

    # Schedule
    scheduled_date: Optional[datetime] = None
    actual_date: Optional[datetime] = None
    duration_hours: float = 2.0

    # Results
    status: str = "scheduled"  # scheduled, in_progress, completed, cancelled
    outcome: str = ""  # approved, approved_with_actions, rejected
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    meeting_notes: str = ""

    def add_action_item(self, description: str, assigned_to: str, due_date: datetime):
        """Add an action item from the review."""
        self.action_items.append({
            'id': str(uuid.uuid4())[:8],
            'description': description,
            'assigned_to': assigned_to,
            'due_date': due_date.isoformat(),
            'status': 'open'
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'review_type': self.review_type,
            'description': self.description,
            'reviewers': self.reviewers,
            'presenter': self.presenter,
            'components_in_scope': self.components_in_scope,
            'documents': self.documents,
            'scheduled_date': self.scheduled_date.isoformat() if self.scheduled_date else None,
            'actual_date': self.actual_date.isoformat() if self.actual_date else None,
            'duration_hours': self.duration_hours,
            'status': self.status,
            'outcome': self.outcome,
            'action_items': self.action_items,
            'meeting_notes': self.meeting_notes
        }


@dataclass
class ChangeRequest:
    """An engineering change request."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    title: str = ""
    change_type: ChangeType = ChangeType.ENGINEERING_CHANGE
    description: str = ""
    justification: str = ""

    # Requestor
    requested_by: str = ""
    requested_date: datetime = field(default_factory=datetime.now)

    # Affected items
    affected_components: List[str] = field(default_factory=list)
    affected_documents: List[str] = field(default_factory=list)

    # Impact assessment
    cost_impact: float = 0.0
    schedule_impact_days: int = 0
    quality_impact: str = ""
    risk_assessment: str = ""

    # Workflow
    status: str = "draft"  # draft, submitted, in_review, approved, rejected, implemented
    approvers: List[Dict[str, Any]] = field(default_factory=list)
    implementation_plan: str = ""
    implementation_date: Optional[datetime] = None

    # Attachments
    attachments: List[str] = field(default_factory=list)

    def add_approver(self, stakeholder_id: str, role: str):
        """Add an approver to the change request."""
        self.approvers.append({
            'stakeholder_id': stakeholder_id,
            'role': role,
            'status': 'pending',
            'decision_date': None,
            'comments': ''
        })

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'title': self.title,
            'change_type': self.change_type.name,
            'description': self.description,
            'justification': self.justification,
            'requested_by': self.requested_by,
            'requested_date': self.requested_date.isoformat(),
            'affected_components': self.affected_components,
            'affected_documents': self.affected_documents,
            'cost_impact': self.cost_impact,
            'schedule_impact_days': self.schedule_impact_days,
            'quality_impact': self.quality_impact,
            'risk_assessment': self.risk_assessment,
            'status': self.status,
            'approvers': self.approvers,
            'implementation_plan': self.implementation_plan,
            'implementation_date': self.implementation_date.isoformat() if self.implementation_date else None,
            'attachments': self.attachments
        }


class CollaborationHub:
    """
    Central hub for multi-discipline collaboration in product development.

    Features:
    - Stakeholder management
    - Task assignment and tracking
    - Design reviews
    - Engineering change management
    - Real-time notifications
    - Activity tracking
    """

    def __init__(self, product_id: str = ""):
        """Initialize the collaboration hub."""
        self.product_id = product_id

        # Data stores
        self.stakeholders: Dict[str, Stakeholder] = {}
        self.tasks: Dict[str, CollaborationTask] = {}
        self.reviews: Dict[str, DesignReview] = {}
        self.change_requests: Dict[str, ChangeRequest] = {}

        # Activity log
        self.activity_log: List[Dict[str, Any]] = []

        # Notifications queue
        self.notifications: List[Dict[str, Any]] = []

    def add_stakeholder(self, stakeholder: Stakeholder) -> str:
        """Add a stakeholder to the hub."""
        self.stakeholders[stakeholder.id] = stakeholder
        self._log_activity('stakeholder_added', stakeholder.id,
                           f"Added stakeholder: {stakeholder.name}")
        return stakeholder.id

    def remove_stakeholder(self, stakeholder_id: str):
        """Remove a stakeholder."""
        if stakeholder_id in self.stakeholders:
            name = self.stakeholders[stakeholder_id].name
            del self.stakeholders[stakeholder_id]
            self._log_activity('stakeholder_removed', stakeholder_id,
                               f"Removed stakeholder: {name}")

    def create_task(self, title: str, description: str, assigned_to: List[str],
                    priority: TaskPriority = TaskPriority.MEDIUM,
                    created_by: str = "") -> CollaborationTask:
        """Create a new collaboration task."""
        task = CollaborationTask(
            title=title,
            description=description,
            assigned_to=assigned_to,
            priority=priority,
            created_by=created_by
        )
        self.tasks[task.id] = task

        # Update stakeholder assignments
        for stakeholder_id in assigned_to:
            if stakeholder_id in self.stakeholders:
                self.stakeholders[stakeholder_id].assigned_tasks.append(task.id)

        self._log_activity('task_created', task.id, f"Created task: {title}")
        self._notify(assigned_to, 'task_assigned', f"You have been assigned task: {title}")

        return task

    def update_task_status(self, task_id: str, status: TaskStatus,
                           percent_complete: int = 0):
        """Update task status."""
        if task_id in self.tasks:
            task = self.tasks[task_id]
            old_status = task.status
            task.status = status
            task.percent_complete = percent_complete

            if status == TaskStatus.COMPLETED:
                task.completed_date = datetime.now()

            self._log_activity('task_updated', task_id,
                               f"Task '{task.title}' status: {old_status.name} -> {status.name}")

    def create_review(self, title: str, review_type: str,
                      reviewers: List[str], presenter: str,
                      components: List[str] = None) -> DesignReview:
        """Create a design review."""
        review = DesignReview(
            title=title,
            review_type=review_type,
            reviewers=reviewers,
            presenter=presenter,
            components_in_scope=components or []
        )
        self.reviews[review.id] = review

        self._log_activity('review_created', review.id, f"Created review: {title}")
        self._notify(reviewers, 'review_scheduled', f"You are invited to review: {title}")

        return review

    def complete_review(self, review_id: str, outcome: str, notes: str = ""):
        """Complete a design review."""
        if review_id in self.reviews:
            review = self.reviews[review_id]
            review.status = 'completed'
            review.outcome = outcome
            review.meeting_notes = notes
            review.actual_date = datetime.now()

            self._log_activity('review_completed', review_id,
                               f"Review '{review.title}' completed: {outcome}")

    def submit_change_request(self, title: str, change_type: ChangeType,
                              description: str, justification: str,
                              requested_by: str,
                              affected_components: List[str] = None) -> ChangeRequest:
        """Submit an engineering change request."""
        cr = ChangeRequest(
            title=title,
            change_type=change_type,
            description=description,
            justification=justification,
            requested_by=requested_by,
            affected_components=affected_components or [],
            status='submitted'
        )
        self.change_requests[cr.id] = cr

        self._log_activity('change_request_submitted', cr.id,
                           f"Change request submitted: {title}")

        # Notify approvers (program managers)
        managers = [s.id for s in self.stakeholders.values()
                    if s.role == StakeholderRole.PROGRAM_MANAGER]
        self._notify(managers, 'change_request', f"New change request requires review: {title}")

        return cr

    def approve_change_request(self, cr_id: str, approver_id: str,
                               approved: bool, comments: str = ""):
        """Approve or reject a change request."""
        if cr_id in self.change_requests:
            cr = self.change_requests[cr_id]
            for approver in cr.approvers:
                if approver['stakeholder_id'] == approver_id:
                    approver['status'] = 'approved' if approved else 'rejected'
                    approver['decision_date'] = datetime.now().isoformat()
                    approver['comments'] = comments
                    break

            # Check if all approvers have decided
            all_decided = all(a['status'] != 'pending' for a in cr.approvers)
            all_approved = all(a['status'] == 'approved' for a in cr.approvers)

            if all_decided:
                cr.status = 'approved' if all_approved else 'rejected'
                self._log_activity('change_request_decided', cr_id,
                                   f"Change request '{cr.title}': {cr.status}")

    def get_stakeholder_workload(self, stakeholder_id: str) -> Dict[str, Any]:
        """Get workload summary for a stakeholder."""
        if stakeholder_id not in self.stakeholders:
            return {}

        stakeholder = self.stakeholders[stakeholder_id]
        tasks = [self.tasks[tid] for tid in stakeholder.assigned_tasks
                 if tid in self.tasks]

        return {
            'stakeholder': stakeholder.name,
            'total_tasks': len(tasks),
            'by_status': {
                status.name: len([t for t in tasks if t.status == status])
                for status in TaskStatus
            },
            'by_priority': {
                priority.name: len([t for t in tasks if t.priority == priority])
                for priority in TaskPriority
            },
            'overdue': len([t for t in tasks
                            if t.due_date and t.due_date < datetime.now()
                            and t.status not in [TaskStatus.COMPLETED, TaskStatus.CANCELLED]])
        }

    def get_project_dashboard(self) -> Dict[str, Any]:
        """Get project-level dashboard data."""
        all_tasks = list(self.tasks.values())
        all_reviews = list(self.reviews.values())
        all_crs = list(self.change_requests.values())

        return {
            'product_id': self.product_id,
            'stakeholder_count': len(self.stakeholders),
            'tasks': {
                'total': len(all_tasks),
                'completed': len([t for t in all_tasks if t.status == TaskStatus.COMPLETED]),
                'in_progress': len([t for t in all_tasks if t.status == TaskStatus.IN_PROGRESS]),
                'blocked': len([t for t in all_tasks if t.status == TaskStatus.BLOCKED]),
                'critical': len([t for t in all_tasks if t.priority == TaskPriority.CRITICAL])
            },
            'reviews': {
                'total': len(all_reviews),
                'scheduled': len([r for r in all_reviews if r.status == 'scheduled']),
                'completed': len([r for r in all_reviews if r.status == 'completed']),
                'approved': len([r for r in all_reviews if r.outcome == 'approved'])
            },
            'change_requests': {
                'total': len(all_crs),
                'pending': len([c for c in all_crs if c.status in ['submitted', 'in_review']]),
                'approved': len([c for c in all_crs if c.status == 'approved']),
                'rejected': len([c for c in all_crs if c.status == 'rejected'])
            },
            'recent_activity': self.activity_log[-10:] if self.activity_log else []
        }

    def _log_activity(self, activity_type: str, item_id: str, description: str):
        """Log an activity."""
        self.activity_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': activity_type,
            'item_id': item_id,
            'description': description
        })

    def _notify(self, stakeholder_ids: List[str], notification_type: str, message: str):
        """Queue a notification."""
        for sid in stakeholder_ids:
            if sid in self.stakeholders:
                self.notifications.append({
                    'id': str(uuid.uuid4())[:8],
                    'stakeholder_id': sid,
                    'type': notification_type,
                    'message': message,
                    'timestamp': datetime.now().isoformat(),
                    'read': False
                })

    def get_notifications(self, stakeholder_id: str, unread_only: bool = True) -> List[Dict]:
        """Get notifications for a stakeholder."""
        notifications = [n for n in self.notifications
                         if n['stakeholder_id'] == stakeholder_id]
        if unread_only:
            notifications = [n for n in notifications if not n['read']]
        return notifications

    def export_state(self) -> Dict[str, Any]:
        """Export the entire collaboration state."""
        return {
            'product_id': self.product_id,
            'exported_date': datetime.now().isoformat(),
            'stakeholders': {k: v.to_dict() for k, v in self.stakeholders.items()},
            'tasks': {k: v.to_dict() for k, v in self.tasks.items()},
            'reviews': {k: v.to_dict() for k, v in self.reviews.items()},
            'change_requests': {k: v.to_dict() for k, v in self.change_requests.items()},
            'activity_log': self.activity_log
        }
