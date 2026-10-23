# Uploading videos to YouTube

`video/tools/upload.mjs` uploads a built video kit (`video/<slug>/out/*.mp4` plus `video/<slug>/youtube/`)
through the YouTube Data API v3. It does every step the API supports. A few steps stay manual because the
API can't do them. The step-by-step manual route is in each kit's `UPLOAD.md`.

## One-time setup

### 1. Google Cloud OAuth client

1. Open [Google Cloud Console](https://console.cloud.google.com/) and create a project, for example
   `vidyalai-youtube`.
2. **APIs & Services → Library** → enable **YouTube Data API v3**.
3. **APIs & Services → OAuth consent screen**:
   - User type: **External**. App name: anything, for example "VidyalAI uploader".
   - Scopes: add none here. The script requests `youtube.upload` and `youtube.force-ssl` when you sign in.
   - **Test users**: add the Google account that owns the YouTube channel. In the newer console this is
     **Google Auth Platform → Audience → Test users → Add users**. If the channel is a Brand Account, add
     the Gmail address you sign in with, not the channel's name. Check that the address then
     appears in the list.
   - Leave the app in **Testing**. You don't need to publish it or have it verified.
4. **APIs & Services → Credentials → Create credentials → OAuth client ID** → application type
   **Desktop app** → **Download JSON**.
5. Save the file as `~/.config/vidyalai/youtube-client.json`:

   ```sh
   mkdir -p ~/.config/vidyalai && chmod 700 ~/.config/vidyalai
   mv ~/Downloads/client_secret_*.json ~/.config/vidyalai/youtube-client.json
   ```

   On WSL the Windows Downloads folder is `/mnt/c/Users/<you>/Downloads/`.

Credentials and tokens live only in `~/.config/vidyalai/`, never in the repo.

### 2. YouTube Studio upload defaults

The API can't set the following fields on a single video, so set them as channel defaults instead. Go to
**Studio → Settings → Upload defaults**.

| Tab | Setting |
|---|---|
| Basic info | Visibility: Private (the script sets privacy itself, but this is a safe default) |
| Advanced settings | **License**: Standard YouTube License. **Category**: Education. **Video language**: English. **Caption certification**: "This content has never aired on television in the U.S." |
| Advanced settings | **Automatic chapters**, **Featured places**, **Automatic concepts**: all unchecked |
| Advanced settings | **Comments**: on, **Hold potentially inappropriate comments for review → Increase strictness** (Strict) |

Age restriction (none) and paid promotion (No) are YouTube's defaults, so you don't need to change them.

After the first upload, open one video in Studio and check that these settings were applied. API uploads
may not pick up every default; if one is missing, set it by hand on each video.

### 3. Channel prerequisites

- **Phone verification** at <https://www.youtube.com/verify>. Custom thumbnails need it.
- Each series needs its own playlist, named exactly like the series in `src/data/series.mjs`: **LLM System Design**,
  **LLM Fundamentals**, **LLM Inference & Optimization**. Create it in Studio before the series' first upload; the
  script looks it up by its exact title (or by `youtube.playlist` in `script.yaml`, if set). Then add its URL as
  `playlist` in `src/data/series.mjs` so the series page and article videos link to it.

### 4. Sign in

```sh
node video/tools/upload.mjs auth
```

Open the link it prints, sign in with the channel's account, and allow access. You'll see "Google hasn't
verified this app": choose **Continue**, because it's your own app. The token is saved to
`~/.config/vidyalai/youtube-token.json`.

If Google shows "Error 403: access_denied … has not completed the Google verification process", the account
you signed in with isn't on the test-user list. Add it (step 1.3), then run `auth` again. If several Google
accounts are signed in to the browser, pick the one you added.

In Testing mode Google expires the sign-in after **7 days**. When a command fails with `invalid_grant`, run
`auth` again.

## Each release

```sh
# 0. Build the videos (see the article-to-video skill)
node video/tools/build.mjs video/<slug>

# 1. Check what will be sent, without calling the API
node video/tools/upload.mjs upload video/<slug> --dry-run

# 2. Upload everything as private: main video first, then the Shorts
node video/tools/upload.mjs upload video/<slug>
```

For each video, `upload` sets:
- the title, description (its timestamps become the chapters) and tags,
- category Education and language English,
- "not made for kids" and AI disclosure Yes,
- the captions and the playlist,
- the thumbnail (main video only).

It saves its progress in `video/<slug>/youtube/uploaded.json` after every step. If it stops, run it again:
it resumes from where it stopped and never uploads a video twice. Commit `uploaded.json`, because it
records the video IDs.

Then do the manual steps in Studio. `upload` prints a link for each.

3. **End screen** on the main video, from the end-card start time (`youtube/main/endcard.txt`, in
   m:ss:frames format). Put Subscribe in the right-hand circle and the playlist in the left-hand frame. If
   the next video is already public, use that video instead of the playlist.
4. **Related video** on each Short → the main video.
5. **Checks** tab: wait until the copyright check finishes.

```sh
# 6. Check processing, then publish: main video first, then the Shorts
node video/tools/upload.mjs status  video/<slug>
node video/tools/upload.mjs publish video/<slug>
```

`publish` refuses to run while any video is still processing. It then makes each video public and posts
the comment from `youtube/main/pinned-comment.txt`.

7. **Pin the comment** in Studio, using the link `publish` prints.
8. **Embed the video** in the article (`video:` frontmatter plus the thumbnail; see the article-to-video
   skill), then commit and deploy.
9. Swap the previous video's end-screen **playlist** element for this new video.

### Options

- `--only main,short-id` limits `upload` or `publish` to the videos you list.
- To redo a video, delete it in Studio, remove its entry from `uploaded.json`, and run `upload` again.

## Daily limits

The API has two separate daily quotas per Google Cloud project. Both reset at midnight Pacific Time, which
is 12:30 or 1:30 PM IST depending on US daylight saving time.

| Quota | Daily limit | What uses it |
|---|---|---|
| Video uploads | 100 uploads | `videos.insert` (1 per video) |
| Everything else | 10,000 units | The calls below |

What each video costs from the 10,000 units:

| Call | Units | Main video | Short |
|---|---|---|---|
| `captions.insert` | 400 | 400 | 400 |
| `thumbnails.set` | 50 | 50 | - |
| `playlistItems.insert` | 50 | 50 | 50 |
| `videos.update` (publish) | 50 | 50 | 50 |
| `commentThreads.insert` | 50 | 50 | - |
| **Total** | | **600** | **500** |

Lookups (`playlists.list`, `videos.list` in `status` and `publish`) cost 1 unit each.

| Upload pattern | Units | Per day |
|---|---|---|
| One release (1 main video and 3 Shorts) | about 2,100 | **4 releases** (16 videos, about 8,400 units) |
| Shorts only | about 500 each | about **19 Shorts** |
| Main videos only | about 600 each | about **16 main videos** |

Captions use most of the quota. Re-running `upload` doesn't spend quota on steps that already succeeded.

YouTube also limits how many videos a **channel** can upload in 24 hours. YouTube doesn't publish this
number: it varies by channel, and it's lower for new or unverified channels. If you hit it, uploads fail
with `uploadLimitExceeded`. Wait 24 hours and re-run `upload`: it resumes. For one release a day, neither
limit will get in your way.

Current limits: [quota costs](https://developers.google.com/youtube/v3/determine_quota_cost) and
[`videos.insert`](https://developers.google.com/youtube/v3/docs/videos/insert). They were checked on
2026-10-21. If you ever need more quota, apply with the
[YouTube API Services audit and quota extension form](https://support.google.com/youtube/contact/yt_api_form).
