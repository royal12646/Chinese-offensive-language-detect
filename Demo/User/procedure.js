const express = require('express')
const app = express()
const cors = require('cors')
const multer = require('multer')
const path = require('path')
const mysql = require('mysql')
const bodyParser = require('body-parser');

app.use(cors())
app.use(bodyParser.urlencoded({ extended: true }));

const db = mysql.createConnection({
    host: '172.22.235.92',
    user: 'royal_database',
    password: 'fKHb5ejBhCKdjZtH',
    database: 'royal_database',
    port: 3306
});

db.connect((err) => {
    if (err) {
        console.log('数据库连接失败', err);
        return;
    }
    console.log('数据库连接成功');
});

app.post('/api/login', (req, res) => {
    const { username, password } = req.body;
    const sql = 'SELECT * FROM users WHERE username = ? AND password = ?';

    db.query(sql, [username, password], (err, results) => {
        if (err) {
            console.log(err);
            return res.status(500).json({
                status: 500,
                message: '数据库查询失败'
            });
        }

        console.log('Query results:', results);

        if (results.length > 0) {
            res.status(200).json({
                status: 200,
                message: '登录成功',
                data: results
            });
        } else {
            res.status(401).json({
                status: 401,
                message: '用户名或密码错误'
            });
        }
    });
});

app.post('/api/register', (req, res) => {
    const { username, password, email, captchaInput } = req.body;
    console.log('Received Data:', req.body); // 输出接收到的数据


    // 检查用户名、密码和邮箱是否为空
    if (!username || !password || !email) {
        return res.status(400).send({
            status: 400,
            message: '请填写完整的注册信息'
        });
    }

    // 插入数据到数据库
    const sql = 'INSERT INTO users (username, password, email) VALUES (?, ?, ?)';
    db.query(sql, [username, password, email], (err) => {
        if (err) {
            console.error('数据库插入失败:', err);
            return res.status(500).send({
                status: 500,
                message: '数据库插入失败'
            });
        }
        res.status(200).send({
            status: 200,
            message: '注册成功'
        });
    });
});

app.get('/api/userinfo', (req, res) => {
    const { email } = req.query;

    if (!email) {
        return res.status(400).json({
            status: 400,
            message: '邮箱地址是必需的'
        });
    }

    const sql = 'SELECT * FROM users WHERE email = ?';

    db.query(sql, [email], (err, results) => {
        if (err) {
            console.log(err);
            return res.status(500).json({
                status: 500,
                message: '数据库查询失败'
            });
        }

        console.log('Query results:', results);

        if (results.length > 0) {
            res.status(200).json({
                status: 200,
                message: '用户信息获取成功',
                data: results[0]
            });
        } else {
            res.status(404).json({
                status: 404,
                message: '用户未找到'
            });
        }
    });
});


app.listen(8999, () => {
    console.log('Server is running on http://172.22.235.92:8999');
});

app.use(express.json());
// 静态文件服务
app.use(express.static('uploads'));

// 配置 multer
const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads/');
    },
    filename: (req, file, cb) => {
        const uniqueSuffix = Date.now() + '-' + Math.round(Math.random() * 1E9);
        cb(null, file.fieldname + '-' + uniqueSuffix + path.extname(file.originalname));
    }
});

const upload = multer({ storage: storage });

// 处理头像上传
app.post('/api/upload-avatar', upload.single('file'), (req, res) => {
    const file = req.file;
    if (!file) {
        return res.status(400).send('No file uploaded.');
    }
    const fileUrl = `http://172.22.235.92:9999/${file.filename}`;
    console.log('File URL:', fileUrl); // 调试输出
    res.json({ url: fileUrl });
});

app.post('/api/update-user-profile', (req, res) => {
    const { username, avatar, gender, password, email, introduction, id } = req.body; // 从请求体中获取id
    console.log(req.body)
    if (!id) {
        return res.status(400).send('用户ID丢失');
    }

    const query = `
    UPDATE users
    SET username = ?, avatar = ?, gender = ?, password = ?, email = ?,introduction=?
    WHERE id = ?
    `;

    db.query(query, [username, avatar, gender, password, email, introduction, id], (error) => {
        if (error) {
            console.error('Database update failed:', error); // 详细记录错误信息
            return res.status(500).send('Database update failed.');
        }
        res.send('Profile updated successfully.');
    });
});
// 静态文件服务
app.use(express.static('uploads'));

// 获取所有用户数据的接口
app.get('/api/getUserData', (req, res) => {
    const sql = 'SELECT * FROM users';

    db.query(sql, (err, results) => {
        if (err) {
            console.log(err);
            return res.status(500).json({
                status: 500,
                message: '数据库查询失败'
            });
        }

        if (results.length > 0) {
            res.status(200).json({
                status: 200,
                message: '用户数据获取成功',
                data: results
            });
        } else {
            res.status(404).json({
                status: 404,
                message: '没有用户数据'
            });
        }
    });
});

// 检查用户名是否已存在
app.post('/api/checkUsername', (req, res) => {
    const { username } = req.body;

    const sql = "SELECT COUNT(*) AS count FROM users WHERE username = ?";
    db.query(sql, [username], (err, result) => {
        if (err) {
            return res.status(500).json({ error: "服务器错误，请稍后再试。" });
        }

        const count = result[0].count;
        if (count > 0) {
            res.json({ exists: true }); // 用户名已存在
        } else {
            res.json({ exists: false }); // 用户名不存在
        }
    });
});
