# Privacy Policy (Template)

**This is a template for the facility deploying this software to adapt and publish, not a policy Anthropic, this project's authors, or any software vendor are issuing on your behalf. It is not legal advice.** Before using it, have it reviewed by counsel familiar with your jurisdiction and, if you are a healthcare or care facility, with the health-privacy laws that apply to you (HIPAA in the US, GDPR in the EU/UK, and any state or local eldercare regulations). Replace every `[bracketed]` placeholder before publishing.

Last updated: [date]

## Who this covers

This policy applies to the Fight Detection System deployed by **[Facility Name]** ("the Facility") to monitor common areas for resident and patient safety. The Facility operates this software and is the data controller. This is self-hosted software; there is no third-party vendor receiving or processing your data unless the Facility has separately configured one (for example, an email provider used to send alerts).

## What this system collects

- **Video feeds** from cameras installed in the areas the Facility has designated for monitoring. Video is analyzed in real time; frames are not permanently stored except when an incident triggers a recorded clip (see below).
- **Incident clips**: a short video (roughly 10 seconds before and 15 seconds after a detected incident) is saved when the system detects a possible physical altercation.
- **Incident metadata**: timestamp, camera and room identifiers, a confidence score, and any notes a caregiver adds during review.
- **Caregiver account data**: name, email address, and a securely hashed password for staff who log into the dashboard. Plaintext passwords are never stored.
- **Caregiver-entered content**: shift notes, profile fields, and incident review notes that caregivers choose to enter.

This system is not designed to, and should not be configured to, monitor private areas such as resident bedrooms, bathrooms, or changing areas, unless the Facility has an independent legal basis and resident/guardian consent for doing so.

## Why this data is collected

Solely to detect and respond to physical altercations or safety incidents in common areas, so staff can intervene quickly. Incident clips and metadata exist to let staff review what happened, confirm or dispute a detection, and maintain a record for safety follow-up.

## Who can access this data

Only caregiver and administrator accounts provisioned by the Facility's system administrator can log in and view camera feeds, incident clips, or incident metadata. There is no public or self-service access. Administrators additionally have access to system-level configuration (detection thresholds, the underlying detection model).

## How long data is kept

[The Facility should state its retention period here — for example: "Incident clips and metadata are retained for [N days/months] and then deleted." This software does not currently auto-delete clips; the Facility is responsible for defining and enforcing a retention schedule, including deleting clips manually or via a scheduled task until automatic retention is implemented.]

## How data is protected

- Dashboard access requires a caregiver login; every page and API endpoint checks that a valid session exists.
- Passwords are hashed, never stored in plaintext.
- Communication between the detection service and the dashboard uses a separate internal key, not exposed to the browser.
- The Facility is responsible for serving the dashboard over HTTPS in any deployment reachable outside a trusted local network, and for restricting network access to the server running this software.

## Residents' and patients' rights

[The Facility should describe, per its jurisdiction's requirements: how residents/patients or their legal guardians are notified that common areas are monitored, how they can request information about an incident involving them, and how they can raise a concern or complaint about this monitoring.]

## Changes to this policy

[The Facility should describe how it will notify staff, residents, and guardians of material changes to this policy.]

## Contact

[Facility contact information for privacy questions.]
