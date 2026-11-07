# Northwind Cloud Security Overview

Security is a shared responsibility. Northwind Cloud secures the underlying infrastructure, and customers are responsible for how they configure accounts, networks and applications.

## Data encryption

All customer data is encrypted at rest with AES-256. Data in transit between clients and Northwind Cloud endpoints is protected with TLS 1.2 or higher, and older protocol versions are rejected. Enterprise customers can bring their own encryption keys through the key management service.

## Access control

Multi-factor authentication is required for every console user on Business and Enterprise plans, and is strongly recommended on the Free plan. Organisation admins can enforce single sign-on through SAML or OIDC identity providers. API tokens are scoped to a single project and expire after 90 days unless they are rotated.

## Compliance

Northwind Cloud is certified against ISO 27001 and completes a SOC 2 Type II audit every year. Audit reports are available to customers under NDA from the Trust Center in the console.

## Vulnerability disclosure

Security researchers can report vulnerabilities to security@northwind.example. We acknowledge reports within 48 hours and run a public bug bounty programme for qualifying findings.

## Data residency

Customers choose a region when they create a project, and data stays in that region unless the customer configures cross-region replication. Regions are currently available in Frankfurt, Dublin, Virginia, Oregon and Singapore.
