# TRAM Docker Images

## Overview

See the [main README](../README.md) for an overview of installing TRAM via
Docker. This document contains some additional detail that may be useful for
customizing your TRAM instance.

## Environment Variables

<table>
  <thead>
    <tr>
      <th>Variable</th>
      <th>Required<th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>ALLOWED_HOSTS</code></td>
      <td>Yes<td>
      <td>A list of hostnames that TRAM can be served from.</td>
    </tr>
    <tr>
      <td><code>DJANGO_SUPERUSER_USERNAME</code></td>
      <td>Yes<td>
      <td>The username for the TRAM super user (the default account you sign in with).</td>
    </tr>
    <tr>
      <td><code>DJANGO_SUPERUSER_PASSWORD</code></td>
      <td>Yes<td>
      <td>The password for the TRAM super user.</td>
    </tr>
    <tr>
      <td><code>DJANGO_SUPERUSER_EMAIL</code></td>
      <td>Yes<td>
      <td>The email address for the TRAM super user. (Not used in pratice, doesn't need to be a real address.)</td>
    </tr>
    <tr>
      <td><code>DATA_DIRECTORY</code></td>
      <td>No<td>
      <td>Any ML data and DB data is stored at the path indicated at this environment variable. Defaults to <code>./data</code>.</td>
    </tr>
    <tr>
      <td><code>SECRET_KEY</code></td>
      <td>No<td>
      <td>
        A cryptographic secret used by Django. This secret can be generated using this command:
        <code>$ python3 -c "import secrets; print(secrets.token_urlsafe())"</code>
        If not provided, then a random secret is created at startup.
      </td>
    </tr>
    <tr>
      <td><code>DEBUG</code></td>
      <td>No<td>
      <td>Set to `true` or `yes` to enable Django debug mode, otherwise debug mode is disabled.</td>
    </tr>
  </tbody>
</table>
