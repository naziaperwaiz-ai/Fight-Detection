# src/auth/create_caregiver.py
#
# CLI for provisioning accounts. This is the only way to create the very
# first account (there's no admin yet to issue an invite). After that,
# admins can invite additional caregivers/admins from the dashboard's
# System Settings -> Team access panel (see auth/users.py's create_invite);
# this CLI still works too, if you'd rather script account creation.
#
# Two roles exist: "caregiver" (default) and "admin". Admins can change
# the detection model and detection defaults from System Settings;
# caregivers see those as read-only.
#
# Usage (run from the project root, or from src/):
#   python -m auth.create_caregiver --email jane@ward.org --name "Jane Doe"
#   python -m auth.create_caregiver --email lead@ward.org --name "Shift Lead" --role admin
#   python -m auth.create_caregiver --list
#   python -m auth.create_caregiver --delete jane@ward.org
#
# If --password is omitted you'll be prompted via getpass so the password
# never appears in shell history.

import argparse
import getpass
import sys

from auth.users import create_caregiver, list_caregivers, delete_caregiver, set_assigned_rooms


def main():
    parser = argparse.ArgumentParser(description="Manage caregiver and admin accounts.")
    parser.add_argument("--email", help="Account email (used as login).")
    parser.add_argument("--name", help="Display name.")
    parser.add_argument("--password", help="Password (omit to be prompted).")
    parser.add_argument(
        "--role", choices=["caregiver", "admin"], default="caregiver",
        help="Account role. Admins can edit the model/detection defaults; caregivers can't. Default: caregiver.",
    )
    parser.add_argument(
        "--rooms", metavar="ROOM1,ROOM2",
        help="Comma-separated rooms this caregiver can see cameras/incidents/clips for. "
             "Ignored for admins (admins always see every room). A caregiver created with "
             "no --rooms sees NOTHING until an admin assigns rooms -- this is deliberate, "
             "not a bug: default access to patient video should never be 'everything'.",
    )
    parser.add_argument("--list", action="store_true", help="List existing accounts.")
    parser.add_argument("--delete", metavar="EMAIL", help="Delete an account by email.")
    parser.add_argument(
        "--set-rooms", metavar="EMAIL",
        help="Update an EXISTING account's assigned rooms (use with --rooms; omit --rooms to clear all access).",
    )
    args = parser.parse_args()

    if args.list:
        users = list_caregivers()
        if not users:
            print("No accounts yet.")
            return
        for u in users:
            rooms = u.get("assigned_rooms", [])
            room_note = "all rooms (admin)" if u.get("role") == "admin" else (", ".join(rooms) if rooms else "NO ROOMS ASSIGNED -- sees nothing")
            print(f"{u['email']}  ({u.get('name', '')})  role={u.get('role', 'caregiver')}  rooms={room_note}  created {u.get('created_at', '?')}")
        return

    if args.set_rooms:
        rooms = [r.strip() for r in (args.rooms or "").split(",") if r.strip()]
        updated = set_assigned_rooms(args.set_rooms, rooms)
        if updated:
            print(f"Updated {args.set_rooms}: rooms={rooms or '(none)'}")
        else:
            print(f"No account found with email {args.set_rooms}.", file=sys.stderr)
            sys.exit(1)
        return

    if args.delete:
        ok = delete_caregiver(args.delete)
        print(f"Deleted {args.delete}." if ok else f"No account found with email {args.delete}.")
        return

    if not args.email:
        parser.error("--email is required to create an account (or use --list / --delete).")

    password = args.password or getpass.getpass("Password (min 8 chars): ")
    rooms = [r.strip() for r in (args.rooms or "").split(",") if r.strip()]

    try:
        record = create_caregiver(args.email, password, args.name, role=args.role, assigned_rooms=rooms)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Created {record['role']}: {record['email']} (id={record['id']})")


if __name__ == "__main__":
    main()
