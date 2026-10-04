GREEN READER - UPDATE CHECKLIST

FILES IN THIS FOLDER (same layout as your repository)
  app.py, putt_engine.py, gps_mode.py, gps_reader.py, requirements.txt   -> main folder, next to each other
  gps_component/index.html                                              -> a folder called gps_component
  assets/hopewell_valley/  (18 images + scales.json)                    -> new course
  assets/mercer_oaks_east/scales.json                                   -> adds hole 5's width (13.5 yd) to your existing East folder

STEPS (GitHub website)
  1. Unzip this file on your computer.
  2. Open your repository on github.com. Click Add file > Upload files.
  3. Drag in app.py, putt_engine.py, gps_mode.py, gps_reader.py and requirements.txt. They replace the old ones. Commit.
  4. Add file > Create new file. Type  gps_component/index.html  as the name (the slash makes the folder), paste in the contents
     of gps_component/index.html, and Commit.
  5. Open the assets folder. Add file > Upload files. Drag in the 19 files that are INSIDE hopewell_valley
     (do not upload the zip itself), wait until every file shows as uploaded, and Commit.
     Make sure they end up in assets/hopewell_valley/ (if they landed in assets/ directly, move them).
  6. Open assets/mercer_oaks_east. Add file > Create new file. Name it  scales.json  and paste:
        {"5": {"width_yd": 13.5}}
     Commit. (If a scales.json already exists there, edit it instead.)
  7. Delete an old requirements.py if one is still in the main folder.
  8. Wait a minute or two. If the app doesn't update, use Manage app > Reboot app.

CHECK IT WORKED
  - Open the app. Under "Details & comparison" the "Courses found" line should list Hopewell Valley (18 holes) and your Mercer Oaks courses.
  - Hopewell hole 11 should say 18 yd wide by 26 deep. Mercer Oaks East hole 5 should show a putt length near 5 ft for a 5 ft putt.
  - If the app says a code file is out of date, it names the file. Upload that file again.
  - If the deploy fails mentioning streamlit-geolocation, delete that line from requirements.txt and commit.

ON YOUR PHONE
  - Use Safari. Settings > Privacy & Security > Location Services > Safari Websites: While Using the App, Precise Location ON.
  - Pull down to refresh the page after an update.
