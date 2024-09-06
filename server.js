const express = require('express');
const multer = require('multer');
const sqlite3 = require('sqlite3').verbose();
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware to parse incoming JSON requests and form data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static files from the "assets" directory
app.use('/assets', express.static(path.join(__dirname, 'assets')));

// Serve static files directly from root if needed
app.use(express.static(__dirname));

// Configure multer for file upload
const storage = multer.diskStorage({
    destination: function(req, file, cb) {
        cb(null, 'uploads/');
    },
    filename: function(req, file, cb) {
        cb(null, file.originalname);
    }
});
const upload = multer({ 
    storage: storage,
    fileFilter: function(req, file, cb) {
        if (file.mimetype === 'application/octet-stream' && file.originalname.endsWith('.ged')) {
            cb(null, true);
        } else {
            cb(new Error('Only .ged files are allowed!'));
        }
    }
});

// Create SQLite database
const db = new sqlite3.Database('database.sqlite');

db.serialize(() => {
    db.run("CREATE TABLE IF NOT EXISTS files (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, filename TEXT, filepath TEXT)");
    db.run("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT)");
});

// Parse the GEDCOM file (add this function)
function parseGedcomFile(filepath) {
    const fileContent = fs.readFileSync(filepath, 'utf-8');
    const lines = fileContent.split('\n');
    return parseGedcomLines(lines);
}

// Example GEDCOM parser that extracts family data
function parseGedcomLines(lines) {
    const individuals = {};
    const families = {};
    let currentIndividual = null;
    let currentFamily = null;

    lines.forEach(line => {
        const parts = line.trim().split(' ');
        const level = parts[0];
        const tag = parts[1];
        const value = parts.slice(2).join(' ');

        if (level === '0') {
            if (tag.startsWith('@I')) {
                currentIndividual = { id: tag, name: '', events: [], families: [] };
                individuals[tag] = currentIndividual;
            } else if (tag.startsWith('@F')) {
                currentFamily = { id: tag, husband: null, wife: null, children: [] };
                families[tag] = currentFamily;
            }
        } else if (level === '1' && currentIndividual) {
            if (tag === 'NAME') {
                currentIndividual.name = value;
            } else if (tag === 'FAMC' || tag === 'FAMS') {
                currentIndividual.families.push(value);
            }
        } else if (level === '1' && currentFamily) {
            if (tag === 'HUSB') {
                currentFamily.husband = value;
            } else if (tag === 'WIFE') {
                currentFamily.wife = value;
            } else if (tag === 'CHIL') {
                currentFamily.children.push(value);
            }
        }
    });

    // Example output data structure
    return {
        "name": individuals["@I1@"].name || "John /Doe/",
        "children": [
            {
                "name": individuals["@I2@"].name || "Jane /Doe/",
                "children": [
                    { "name": "William /Doe/" },
                    { "name": "Lucy /Doe/" }
                ]
            },
            {
                "name": individuals["@I3@"].name || "James /Doe/",
                "children": []
            }
        ]
    };
}

// Handle GEDCOM file upload and parse it
app.post('/upload', upload.single('file'), (req, res) => {
    const gedcomFilePath = req.file.path;
    const familyData = parseGedcomFile(gedcomFilePath);

    // Send the parsed family data to the frontend
    res.json(familyData);
});

// Serve the fan chart HTML page
app.get('/fan-chart', (req, res) => {
    res.sendFile(path.join(__dirname, 'fan-chart.html'));
});

// Serve privacy policy
app.get('/privacy', (req, res) => {
    res.sendFile(path.join(__dirname, 'privacy.html'));
});

// Serve static files from the root directory
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// Handle file upload and save to database
app.post('/upload', upload.single('file'), (req, res) => {
    if (!req.file) {
        return res.status(400).json({ success: false, message: 'No file uploaded or invalid file type' });
    }

    const filename = req.file.originalname;
    const filepath = req.file.path;
    const username = req.body.username;

    db.run("INSERT INTO files (username, filename, filepath) VALUES (?, ?, ?)", [username, filename, filepath], function(err) {
        if (err) {
            return res.status(500).json({ success: false, message: 'Database error' });
        }
        res.json({ success: true });
    });
});

// Check if a username is already taken
app.get('/check-username', (req, res) => {
    const { username } = req.query;

    db.get("SELECT * FROM users WHERE username = ?", [username], (err, row) => {
        if (err) {
            return res.status(500).json({ success: false, message: 'Database error' });
        }
        if (row) {
            // Username is taken
            res.json({ available: false });
        } else {
            // Username is available
            res.json({ available: true });
        }
    });
});

// Handle account deletion
app.post('/delete-account', (req, res) => {
    const { username } = req.body;

    db.serialize(() => {
        // Delete user's files
        db.all("SELECT filepath FROM files WHERE username = ?", [username], (err, rows) => {
            if (err) {
                return res.status(500).json({ success: false, message: 'Database error' });
            }
            
            // Delete files from the filesystem
            rows.forEach(row => {
                const fs = require('fs');
                fs.unlinkSync(row.filepath);
            });

            // Delete from database
            db.run("DELETE FROM files WHERE username = ?", [username], (err) => {
                if (err) {
                    return res.status(500).json({ success: false, message: 'Database error' });
                }

                // Delete user
                db.run("DELETE FROM users WHERE username = ?", [username], (err) => {
                    if (err) {
                        return res.status(500).json({ success: false, message: 'Database error' });
                    }
                    res.json({ success: true });
                });
            });
        });
    });
});

// Start server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});