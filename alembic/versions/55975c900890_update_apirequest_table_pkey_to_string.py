"""Update ApiRequest table pkey to String

Revision ID: 55975c900890
Revises: 11663b52acd8
Create Date: 2026-08-19 12:21:34.820797

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '55975c900890'
down_revision: Union[str, Sequence[str], None] = '11663b52acd8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Goal: change pkey of ApiRequest from INTEGER to VARCHAR(12)
    # Problem: pkey of ApiRequest is also a fkey of Inference, so need to drop it first and recreate it later

    # Drop FK constraint first
    op.drop_constraint('inference_api_request_id_fkey', 'inference', type_='foreignkey')

    # Drop old integer defaults if any
    op.alter_column('api_request', 'request_id', server_default=None)

    # Alter both columns
    op.alter_column('api_request', 'request_id',
                    existing_type=sa.INTEGER(),
                    type_=sa.String(length=12),
                    existing_nullable=False)
    op.alter_column('inference', 'api_request_id',
               existing_type=sa.INTEGER(),
               type_=sa.String(length=12),
               existing_nullable=False)

    # Recreate FK constraint
    op.create_foreign_key('inference_api_request_id_fkey', 'inference', 'api_request', ['api_request_id'], ['request_id'])

def downgrade() -> None:
    """Downgrade schema."""
    # Unlikely for downgrade to work, since string usually cannot convert back to integer

    op.drop_constraint('inference_api_request_id_fkey', 'inference', type_='foreignkey')

    op.alter_column('inference', 'api_request_id',
               existing_type=sa.String(length=12),
               type_=sa.INTEGER(),
               postgresql_using='api_request_id::integer',
               existing_nullable=False)
    op.alter_column('api_request', 'request_id',
               existing_type=sa.String(length=12),
               type_=sa.INTEGER(),
               postgresql_using='request_id::integer',
               existing_nullable=False)

    op.create_foreign_key('inference_api_request_id_fkey', 'inference', 'api_request', ['api_request_id'], ['request_id'])
