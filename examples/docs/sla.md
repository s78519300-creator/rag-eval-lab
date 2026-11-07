# Northwind Cloud Service Level Agreement

This agreement describes the availability commitments for Northwind Cloud compute, managed databases and object storage.

## Uptime commitment

Northwind Cloud guarantees 99.95% monthly uptime for compute instances and managed databases on the Business and Enterprise plans. Object storage is covered by a 99.9% monthly uptime commitment on every paid plan. The Free plan has no uptime guarantee.

Uptime is measured in one-minute intervals by our external monitoring. A minute counts as unavailable when more than 5% of requests to the service fail with server errors.

## Service credits

If monthly uptime falls below the commitment, affected customers receive service credits. Uptime between 99.0% and 99.95% earns a 10% credit on the monthly bill for the affected service. Uptime below 99.0% earns a 25% credit. Credits are capped at 50% of the monthly bill for that service.

To claim a credit, open a support ticket within 30 days of the incident and include the affected resource IDs.

## Exclusions

The SLA does not cover scheduled maintenance announced at least 72 hours in advance, outages caused by customer configuration, or features labelled as beta. Maintenance windows are normally scheduled on Sundays between 02:00 and 05:00 UTC.

## Status and incident reports

Live service status is published at status.northwind.example. For every major incident we publish a written post-incident report within five business days.
