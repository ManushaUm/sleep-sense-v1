# SleepSense Google OAuth Configuration Guide

This guide details the step-by-step instructions for setting up Google Authentication in the **SleepSense** project. It covers configuring your Google Cloud Platform (GCP) Console, creating credentials, linking credentials in the React Native mobile app (via `expo-auth-session`), and setting up native Android standalone APK support.

---

## 1. Google Cloud Console Setup

### Step 1.1: Create a GCP Project
1. Open the [Google Cloud Console](https://console.cloud.google.com/).
2. Click the project dropdown in the top-left header and select **New Project**.
3. Name your project (e.g. `SleepSense`) and click **Create**.

### Step 1.2: Configure the OAuth Consent Screen
Before creating credentials, you must define the user-facing consent screen:
1. Navigate to **APIs & Services** -> **OAuth consent screen** from the left-hand sidebar.
2. Select **External** (accessible to any user with a Google account) and click **Create**.
3. Fill in the required fields:
   * **App name**: `SleepSense`
   * **User support email**: Select your developer email.
   * **Developer contact information**: Enter your email address.
4. Click **Save and Continue**.
5. Under **Scopes**, click **Add or Remove Scopes** and add:
   * `.../auth/userinfo.profile` (Profile details)
   * `.../auth/userinfo.email` (Email address)
   * `openid` (OpenID identity token)
   * `https://www.googleapis.com/auth/calendar.events.readonly` (Manually paste this scope to allow reading calendar events).
6. Under **Test Users**, add the Google emails you will use for testing, as the app is in "Testing" mode.
7. Click **Save and Continue** and confirm the configuration.

---

## 2. Generating OAuth Client Credentials

You need to create specific Client IDs in **APIs & Services** -> **Credentials** depending on whether you are running in Expo Go or as a compiled standalone app.

### Credentials Overview:
| Client ID Type | Usage | How to Create |
| :--- | :--- | :--- |
| **Web Application** | Used for local development, web client, and as the **default proxy `clientId`** in `expo-auth-session`. | * Choose **Web application**.<br>* Add Authorized Redirect URI: `https://auth.expo.io/@your-expo-username/sleepsense-mobile-app` |
| **Android** | Used for compiled standalone Android APKs (`EAS Build`). | * Choose **Android**.<br>* Enter Package Name: `com.manushau.sleepsense`. <br>* Paste SHA-1 fingerprint (see Section 3). |
| **iOS** | Used for compiled iOS standalone builds. | * Choose **iOS**.<br>* Enter Bundle ID: `com.manushau.sleepsense`. |

---

## 3. Retrieving SHA-1 Certificate from EAS Credentials (For APK)

For the native Google Login redirect to work inside a compiled standalone APK, Google must verify the signature of the build.

1. Open your terminal in the `SleepSense-mobile-app` directory.
2. Run the EAS command:
   ```bash
   eas credentials -p android
   ```
3. Look for the **SHA-1 Fingerprint** under the active build profile credentials (usually outputted as `XX:XX:XX:XX...`).
4. Copy this fingerprint.
5. In your GCP Console, go to **Credentials** -> **Create Credentials** -> **OAuth client ID** -> Select **Android**.
6. Set:
   * **Package Name**: `com.manushau.sleepsense` (must match the `android.package` inside `app.json`).
   * **SHA-1 certificate fingerprint**: Paste the SHA-1 copied from EAS.
7. Click **Create**.

---

## 4. Resolving the Redirect Loop (Expo Go vs. Standsalone)

### Why Google Auth is Failing (The Redirect Loop)
Currently, your GCP Web Client ID has the redirect URI set to:
`https://auth.expo.io/@anonymous/sleepsense-mobile-app`

And `LoginScreen.js` hardcodes the same `@anonymous` redirect.
However, because your mobile project is initialized under your actual Expo username (linked to EAS project ID under account `@manusha_01`), the local app on your phone is listening for callbacks matching your namespace (`@manusha_01`).
1. Google completes 2FA and redirects to the `@anonymous` proxy page.
2. The `@anonymous` proxy page does **not** forward the authentication token to your running app because of the namespace mismatch.
3. The app is left waiting, times out, and redirects you back to the Login screen.

### How to Fix This:
1. Go to your **GCP Console** -> **Credentials**.
2. Click edit (✏️) on **SleepSense Expo Client (Web app)**.
3. Under **Authorized redirect URIs**, add your actual Expo username-based URI:
   * `https://auth.expo.io/@manusha_01/sleepsense-mobile-app`
   *(Replace `manusha_01` with your actual Expo account username if it differs, which you can check by running `eas whoami` in your console).*
4. In `LoginScreen.js`, replace the hardcoded `@anonymous` URI with `makeRedirectUri({ scheme: 'sleepsense' })` as shown in Section 5 below. This automatically resolves to the correct callback namespace during runtime.

---


## 5. React Native Code Implementation

Replace the Google Login initialization hook in [LoginScreen.js](file:///c:/Users/HP/Desktop/Semester%207/AI/Project/SleepSense-mobile-app/src/screens/LoginScreen.js) with this configuration:

```javascript
import * as WebBrowser from 'expo-web-browser';
import * as Google from 'expo-auth-session/providers/google';
import { makeRedirectUri } from 'expo-auth-session';

WebBrowser.maybeCompleteAuthSession();

// Inside LoginScreen:
const [request, response, promptAsync] = Google.useIdTokenAuthRequest({
  // 1. Default Web Client ID (used for Expo Go redirect proxy)
  clientId: '656306996421-8q1shbdao820ekjnhuodjuesmesl4fce.apps.googleusercontent.com',
  
  // 2. Android Client ID (created in Step 3 for standalone builds)
  androidClientId: '656306996421-YOUR_ANDROID_CLIENT_ID.apps.googleusercontent.com',
  
  // 3. Request Calendar scopes
  scopes: [
    'openid',
    'profile',
    'email',
    'https://www.googleapis.com/auth/calendar.events.readonly'
  ],
  
  // 4. Dynamically resolve redirect schema (sleepsense:// on standalone, proxy on Expo Go)
  redirectUri: makeRedirectUri({
    scheme: 'sleepsense',
  }),
});
```
> **Note:** By using `makeRedirectUri` and specifying the `scheme: 'sleepsense'` matching your `app.json`, Expo will automatically direct successful 2FA responses back into the app instead of dropping the connection.
