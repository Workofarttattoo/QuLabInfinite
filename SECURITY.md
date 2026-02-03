# Security Policy

Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**DO NOT** create public GitHub issues for security vulnerabilities.

If you discover a security vulnerability in QuLab Infinite, please report it by emailing:

**security@corporationoflight.com**

Please include the following information:

- Type of vulnerability
- Full description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if available)

### What to Expect

- **Acknowledgment**: Within 48 hours
- **Initial Assessment**: Within 1 week
- **Fix Timeline**: Critical vulnerabilities within 2 weeks
- **Public Disclosure**: After fix is deployed (coordinated disclosure)

## Security Best Practices

### 1. Secrets Management

**NEVER commit secrets to version control:**

```bash
# Generate secure secrets
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

**Required secrets:**
- `JWT_SECRET_KEY` - JWT token signing
- `API_KEY_SALT` - API key hashing
- `SESSION_SECRET` - Session encryption

**Store secrets in:**
- `.env` file (local development)
- Kubernetes Secrets (production)
- HashiCorp Vault (enterprise)
- AWS Secrets Manager (cloud)

### 2. API Key Security

API keys are prefixed with `qlab_` and are:
- 32+ characters random
- Hashed with bcrypt before storage
- Never logged in plain text
- Rotatable without downtime

**Best practices:**
- Generate unique keys per client
- Set appropriate permissions (read/write/admin)
- Monitor usage patterns
- Rotate keys regularly (90 days recommended)
- Revoke unused keys immediately

### 3. Authentication & Authorization

**JWT Tokens:**
- Access tokens expire in 30 minutes
- Refresh tokens expire in 7 days
- Tokens are signed with HS256
- Refresh tokens are single-use only

**Password Security:**
- Minimum 8 characters
- Hashed with bcrypt (cost factor 12)
- Rate limited (5 attempts before lockout)
- Lockout duration: 30 minutes

**Authorization:**
- Role-based access control (RBAC)
- Principle of least privilege
- Regular permission audits

### 4. Rate Limiting

Default limits:
- API requests: 100/minute per IP
- Login attempts: 5/15 minutes per account
- Password reset: 3/hour per account

### 5. Input Validation

All API inputs are validated using Pydantic models:
- Type checking
- Range validation
- SQL injection prevention
- XSS prevention
- SMILES string validation for chemistry

### 6. Database Security

**PostgreSQL:**
- Use parameterized queries (SQLAlchemy)
- Enable SSL connections in production
- Principle of least privilege for DB users
- Regular backups (encrypted at rest)

**Redis:**
- Password protected
- Bind to localhost only
- Enable SSL/TLS in production

### 7. Network Security

**Kubernetes:**
- NetworkPolicies enabled
- Pod-to-pod encryption (service mesh)
- Ingress with TLS termination
- Non-root containers only

**Firewall rules:**
- Allow only necessary ports (80, 443)
- Internal services not exposed
- Rate limiting at ingress

### 8. Dependency Security

**Automated scanning:**
- GitHub Dependabot enabled
- `safety check` in CI/CD
- `bandit` security linting
- Docker image scanning (Trivy)

**Update policy:**
- Critical vulnerabilities: Immediate
- High vulnerabilities: Within 1 week
- Medium/Low: Next release

### 9. Logging & Monitoring

**What we log:**
- Authentication attempts (success/failure)
- API access patterns
- Error conditions
- Security events

**What we DON'T log:**
- Passwords (plain or hashed)
- API keys
- Tokens
- Sensitive personal data

**Log protection:**
- Centralized logging (immutable)
- Access restricted to security team
- Retention: 90 days
- Encrypted at rest and in transit

### 10. Secrets Rotation

**Rotation schedule:**
- JWT secrets: Every 90 days
- API keys: On-demand or every 90 days
- Database passwords: Every 90 days
- SSL/TLS certificates: Before expiration

**Rotation process:**
1. Generate new secret
2. Update Kubernetes Secret
3. Rolling pod restart (zero downtime)
4. Verify functionality
5. Deactivate old secret after grace period

## Security Checklist for Deployment

Before deploying to production:

- [ ] All secrets moved to secure storage (Vault/Secrets Manager)
- [ ] `.env` file in `.gitignore`
- [ ] No hardcoded credentials in code
- [ ] SSL/TLS certificates installed and valid
- [ ] Firewall rules configured
- [ ] Rate limiting enabled
- [ ] Logging and monitoring active
- [ ] Backup strategy implemented
- [ ] Incident response plan documented
- [ ] Security scanning in CI/CD
- [ ] Kubernetes NetworkPolicies applied
- [ ] RBAC configured correctly
- [ ] Non-root containers only
- [ ] Resource limits set
- [ ] Health checks configured

## Known Security Considerations

### Current Limitations

1. **In-memory storage**: Development mode uses in-memory storage for users/API keys. In production, use PostgreSQL with proper backup.

2. **Secret rotation**: Manual process. Consider implementing automatic rotation with HashiCorp Vault.

3. **MFA**: Not currently implemented. Planned for v0.2.0.

4. **API request signing**: Not implemented. Consider for high-security deployments.

5. **Audit logging**: Basic logging implemented. Enhanced audit trail planned for v0.2.0.

## Compliance

QuLab Infinite is designed with security best practices for:
- OWASP Top 10 protection
- HIPAA compliance (for medical labs)
- SOC 2 Type II readiness
- GDPR compliance (data minimization, right to deletion)

## Security Headers

Production deployments should include:

```
Strict-Transport-Security: max-age=31536000; includeSubDomains
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Content-Security-Policy: default-src 'self'
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
```

## Incident Response

In case of security incident:

1. **Immediate**: Isolate affected systems
2. **Within 1 hour**: Assess scope and impact
3. **Within 4 hours**: Notify affected users
4. **Within 24 hours**: Deploy fix or mitigation
5. **Within 72 hours**: Post-mortem and lessons learned

## Security Contacts

- **Security Issues**: security@corporationoflight.com
- **General Support**: support@corporationoflight.com
- **Emergency (Production)**: emergency@corporationoflight.com

## Acknowledgments

We appreciate responsible disclosure and will acknowledge security researchers who report vulnerabilities according to this policy.
